from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from app.utils.constants import CLOUD_TYPES, DATABASE_TYPES, FILE_TYPES


class PolarsDataSourceLoader:
    """Loads source/target DataFrames by connection type using Polars."""

    def __init__(self, schema_cache_ttl_seconds: int = 900):
        self.logger = logging.getLogger("data_compare.loader.polars")
        self.schema_cache_ttl_seconds = schema_cache_ttl_seconds
        self._schema_cache: dict[str, tuple[float, dict[str, str]]] = {}
        self._cache_lock = threading.Lock()

    def load_dataframe(self, side: dict[str, Any]) -> pl.DataFrame:
        start = time.perf_counter()
        connection = side["connection"]
        conn_type = connection["type"]
        schema_def = side.get("schema_def")

        if conn_type in DATABASE_TYPES:
            df = self._load_database(side)
        elif conn_type in FILE_TYPES:
            df = self._load_file(side["uploaded_path"], conn_type, connection)
        elif conn_type in CLOUD_TYPES:
            df = self._load_file(side["file_path"], conn_type, connection)
        else:
            raise ValueError(f"Unsupported connection type: {conn_type}")

        if schema_def:
            df = self._apply_schema(df, schema_def)
        self.logger.info(
            "Polars load dataframe conn_type=%s rows=%s cols=%s elapsed_ms=%.1f",
            conn_type,
            df.height,
            len(df.columns),
            (time.perf_counter() - start) * 1000,
        )
        return df

    def infer_schema(self, side: dict[str, Any]) -> dict[str, str]:
        start = time.perf_counter()
        cache_key = self._schema_cache_key(side)
        now = time.time()
        with self._cache_lock:
            cached = self._schema_cache.get(cache_key)
            if cached and (now - cached[0]) <= self.schema_cache_ttl_seconds:
                return cached[1]

        conn = side["connection"]
        conn_type = conn["type"]
        if conn_type in DATABASE_TYPES:
            schema = self._infer_database_schema(side)
        elif conn_type in FILE_TYPES:
            schema = self._infer_file_schema(side["uploaded_path"], conn_type, conn)
        elif conn_type in CLOUD_TYPES:
            schema = self._infer_file_schema(side["file_path"], conn_type, conn)
        else:
            df = self.load_dataframe(side)
            schema = {name: str(dtype) for name, dtype in df.schema.items()}

        with self._cache_lock:
            self._schema_cache[cache_key] = (now, schema)
        self.logger.info("Polars infer schema conn_type=%s columns=%s elapsed_ms=%.1f", conn_type, len(schema), (time.perf_counter() - start) * 1000)
        return schema

    def _load_database(self, side: dict[str, Any]) -> pl.DataFrame:
        connection = side["connection"]
        conn_type = connection["type"]
        table_name = side["table_name"]
        where_condition = (side.get("where_condition") or "1=1").strip()
        if where_condition.lower().startswith("where "):
            where_condition = where_condition[6:].strip()

        if conn_type == "DATABASE_DUCKDB":
            db_path = connection.get("hostname")
            if not db_path:
                raise ValueError("DuckDB path missing in hostname")
            query = f"SELECT * FROM {table_name} WHERE {where_condition}"
            with duckdb.connect(str(db_path)) as conn:
                return conn.execute(query).pl()

        if conn_type == "DATABASE_DELTA":
            return self._load_delta_table(connection, table_name, where_condition)
        if conn_type == "DATABASE_SNOWFLAKE":
            query = f"SELECT * FROM {table_name} WHERE {where_condition}"
            self.logger.info("Polars snowflake query start table=%s", table_name)
            def run_query() -> pl.DataFrame:
                try:
                    with self._snowflake_connection(connection) as conn:
                        return pl.read_database(query=query, connection=conn)
                except Exception as exc:
                    self._raise_snowflake_optional_dependency_error(exc)
                    raise

            return self._run_with_timeout(
                run_query,
                30,
                "Snowflake query timed out after 30 seconds. Verify hostname/account, auth, and network.",
            )
        if conn_type == "DATABASE_ORACLE":
            query = f"SELECT * FROM {table_name} WHERE {where_condition}"
            self.logger.info("Polars oracle query start table=%s", table_name)
            with self._oracle_connection(connection) as conn:
                return pl.read_database(query=query, connection=conn)
        raise ValueError(f"Unsupported database type: {conn_type}")

    def _load_file(self, path: str, conn_type: str, connection: dict[str, Any]) -> pl.DataFrame:
        if not path:
            raise ValueError("File path is required.")
        if conn_type in FILE_TYPES and not Path(path).exists():
            raise ValueError(f"Local file not found: {path}")
        has_header = bool(connection.get("header", True))
        delimiter = connection.get("field_delim") or ","
        if delimiter == "\\t":
            delimiter = "\t"

        if conn_type.endswith("DELIMITED"):
            return pl.read_csv(path, has_header=has_header, separator=delimiter)
        if conn_type.endswith("EXCEL"):
            return pl.read_excel(path)
        raise ValueError(f"Unsupported file type: {conn_type}")

    def _infer_database_schema(self, side: dict[str, Any]) -> dict[str, str]:
        connection = side["connection"]
        conn_type = connection["type"]
        table_name = side.get("table_name") or ""
        db_name, schema_name, pure_table_name = self._parse_table_parts(table_name, connection)

        if conn_type == "DATABASE_DUCKDB":
            query = (
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                f"WHERE table_schema = '{schema_name}' AND table_name = '{pure_table_name}' "
                "ORDER BY ordinal_position"
            )
            with duckdb.connect(str(connection.get("hostname"))) as conn:
                rows = conn.execute(query).fetchall()
            return {str(col): str(dtype) for col, dtype in rows}

        if conn_type == "DATABASE_DELTA":
            df = self._load_delta_table(connection, table_name, "1=1")
            return {name: str(dtype) for name, dtype in df.schema.items()}
        if conn_type == "DATABASE_SNOWFLAKE":
            query = (
                "SELECT COLUMN_NAME, DATA_TYPE "
                "FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_CATALOG = '{(db_name or connection.get('database_name') or '').upper()}' "
                f"AND TABLE_SCHEMA = '{(schema_name or connection.get('schema_name') or '').upper()}' "
                f"AND TABLE_NAME = '{pure_table_name.upper()}' "
                "ORDER BY ORDINAL_POSITION"
            )
            self.logger.info("Polars snowflake schema query start table=%s", pure_table_name)
            def run_schema_query() -> list[dict[str, Any]]:
                try:
                    with self._snowflake_connection(connection) as conn:
                        return pl.read_database(query=query, connection=conn).to_dicts()
                except Exception as exc:
                    self._raise_snowflake_optional_dependency_error(exc)
                    raise

            rows = self._run_with_timeout(
                run_schema_query,
                30,
                "Snowflake schema query timed out after 30 seconds. Verify hostname/account, auth, and network.",
            )
            return {str(row.get("COLUMN_NAME")): str(row.get("DATA_TYPE")) for row in rows}
        if conn_type == "DATABASE_ORACLE":
            owner = (schema_name or connection.get("schema_name") or "").upper()
            query = (
                "SELECT column_name, data_type "
                "FROM all_tab_columns "
                f"WHERE owner = '{owner}' AND table_name = '{pure_table_name.upper()}' "
                "ORDER BY column_id"
            )
            with self._oracle_connection(connection) as conn:
                rows = pl.read_database(query=query, connection=conn).to_dicts()
            return {str(row.get("COLUMN_NAME", row.get("column_name"))): str(row.get("DATA_TYPE", row.get("data_type"))) for row in rows}
        raise ValueError(f"Unsupported database type: {conn_type}")

    def _infer_file_schema(self, path: str, conn_type: str, connection: dict[str, Any]) -> dict[str, str]:
        df = self._load_file(path, conn_type, connection)
        return {name: str(dtype) for name, dtype in df.schema.items()}

    def _apply_schema(self, df: pl.DataFrame, schema_def: dict[str, str]) -> pl.DataFrame:
        casts: list[pl.Expr] = []
        for col_name, col_type in schema_def.items():
            if col_name in df.columns:
                casts.append(pl.col(col_name).cast(self._to_polars_dtype(col_type), strict=False))
        return df.with_columns(casts) if casts else df

    def _to_polars_dtype(self, dtype_name: str) -> pl.DataType:
        name = str(dtype_name).lower()
        if "int" in name:
            return pl.Int64
        if "double" in name or "float" in name or "decimal" in name:
            return pl.Float64
        if "bool" in name:
            return pl.Boolean
        if "date" in name and "time" not in name:
            return pl.Date
        if "timestamp" in name or "datetime" in name:
            return pl.Datetime
        return pl.Utf8

    def _parse_table_parts(self, table_name: str, connection: dict[str, Any]) -> tuple[str, str, str]:
        parts = [p for p in table_name.split(".") if p]
        db_name = connection.get("database_name", "")
        schema_name = connection.get("schema_name", "main")
        pure_table_name = table_name
        if len(parts) == 3:
            db_name, schema_name, pure_table_name = parts
        elif len(parts) == 2:
            schema_name, pure_table_name = parts
        elif len(parts) == 1:
            pure_table_name = parts[0]
        return db_name, schema_name, pure_table_name

    def _schema_cache_key(self, side: dict[str, Any]) -> str:
        conn = side.get("connection") or {}
        return "|".join(
            [
                str(conn.get("id", "")),
                str(conn.get("type", "")),
                str(side.get("table_name", "")),
                str(side.get("file_path", "")),
                str(side.get("uploaded_path", "")),
                str(conn.get("database_name", "")),
                str(conn.get("schema_name", "")),
            ]
        )

    def _run_with_timeout(self, fn, timeout_seconds: int, timeout_message: str):
        self.logger.info("Run with timeout start timeout_seconds=%s", timeout_seconds)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            result = future.result(timeout=timeout_seconds)
            self.logger.info("Run with timeout completed timeout_seconds=%s", timeout_seconds)
            return result
        except FutureTimeoutError as exc:
            future.cancel()
            self.logger.error("Run with timeout hit timeout_seconds=%s message=%s", timeout_seconds, timeout_message)
            raise ValueError(timeout_message) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _raise_snowflake_optional_dependency_error(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if "255002" in msg and "pyarrow" in msg:
            raise ValueError(
                "Snowflake optional dependency missing: pyarrow. "
                "Install with: pip install pyarrow"
            ) from exc

    def _load_delta_table(self, connection: dict[str, Any], table_name: str, where_condition: str) -> pl.DataFrame:
        try:
            root_path = connection.get("hostname")
            if not root_path:
                raise ValueError("Delta root path missing in hostname")
            table_path = str(Path(root_path) / table_name)
            if not Path(table_path).exists():
                raise ValueError(f"Delta table path not found: {table_path}")
            df = pl.read_delta(table_path)
            condition = (where_condition or "1=1").strip()
            if condition and condition != "1=1":
                # SQL-like filtering for Polars via lazy SQL context.
                return pl.SQLContext(df=df).execute(f"SELECT * FROM df WHERE {condition}").collect()
            return df
        except ModuleNotFoundError as exc:
            raise ValueError("Polars Delta requires 'deltalake' package. Install deltalake to use DATABASE_DELTA.") from exc

    @contextmanager
    def _snowflake_connection(self, connection: dict[str, Any]):
        try:
            import snowflake.connector  # type: ignore
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Polars Snowflake requires 'snowflake-connector-python'. Install it in your environment."
            ) from exc

        host = str(connection.get("hostname") or "").strip()
        host = host.replace("https://", "").replace("http://", "").strip("/")
        account = host.replace(".snowflakecomputing.com", "")
        if not account:
            raise ValueError("Snowflake hostname/account is invalid.")
        kwargs: dict[str, Any] = {
            "account": account,
            "user": connection.get("username"),
            "database": connection.get("database_name"),
            "schema": connection.get("schema_name"),
            # Fast-fail for bad host/account/auth details; prevents long hangs.
            "login_timeout": 20,
            "network_timeout": 20,
            "socket_timeout": 20,
        }
        if connection.get("port"):
            kwargs["port"] = int(connection.get("port"))

        auth_type = (connection.get("authtype") or "").lower()
        if auth_type == "externalbrowser":
            kwargs["authenticator"] = "externalbrowser"
        else:
            kwargs["password"] = connection.get("password")

        try:
            self.logger.info(
                "Polars snowflake connect start account=%s db=%s schema=%s auth=%s",
                account,
                connection.get("database_name"),
                connection.get("schema_name"),
                auth_type or "username/password",
            )
            conn = snowflake.connector.connect(**kwargs)
        except Exception as exc:
            self.logger.exception("Polars snowflake connect failed")
            raise ValueError(f"Snowflake connection failed quickly: {exc}") from exc
        try:
            self.logger.info("Polars snowflake connect success")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _oracle_connection(self, connection: dict[str, Any]):
        try:
            import oracledb  # type: ignore
        except ModuleNotFoundError as exc:
            raise ValueError("Polars Oracle requires 'oracledb'. Install it in your environment.") from exc

        dsn = f"{connection.get('hostname')}:{connection.get('port')}/{connection.get('database_name')}"
        conn = oracledb.connect(
            user=connection.get("username"),
            password=connection.get("password"),
            dsn=dsn,
        )
        try:
            yield conn
        finally:
            conn.close()
