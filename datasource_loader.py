from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.constants import CLOUD_TYPES, DATABASE_TYPES, FILE_TYPES


class DataSourceLoader:
    """Loads source/target DataFrames by connection type."""

    def __init__(
        self,
        spark,
        schema_cache_ttl_seconds: int = 900,
        sample_ratio_small: float = 0.25,
        sample_ratio_medium: float = 0.10,
        sample_ratio_large: float = 0.03,
    ):
        self.logger = logging.getLogger("data_compare.loader.pyspark")
        self.spark = spark
        self.schema_cache_ttl_seconds = schema_cache_ttl_seconds
        self.sample_ratio_small = sample_ratio_small
        self.sample_ratio_medium = sample_ratio_medium
        self.sample_ratio_large = sample_ratio_large
        self._schema_cache: dict[str, tuple[float, dict[str, str]]] = {}
        self._cache_lock = threading.Lock()

    def load_dataframe(self, side: dict[str, Any]) -> DataFrame:
        connection = side["connection"]
        conn_type = connection["type"]
        schema_def = side.get("schema_def")

        df: DataFrame
        if conn_type in DATABASE_TYPES:
            df = self._load_database(side)
        elif conn_type in FILE_TYPES:
            path = side["uploaded_path"]
            df = self._load_file(path, conn_type, connection, schema_def)
        elif conn_type in CLOUD_TYPES:
            path = side["file_path"]
            df = self._load_file(path, conn_type, connection, schema_def)
        else:
            raise ValueError(f"Unsupported connection type: {conn_type}")

        if schema_def:
            df = self._apply_schema(df, schema_def)
        return df

    def infer_schema(self, side: dict[str, Any]) -> dict[str, str]:
        cache_key = self._schema_cache_key(side)
        now = time.time()
        with self._cache_lock:
            cached = self._schema_cache.get(cache_key)
            if cached and (now - cached[0]) <= self.schema_cache_ttl_seconds:
                return cached[1]

        connection = side["connection"]
        conn_type = connection["type"]
        schema: dict[str, str]

        if conn_type in DATABASE_TYPES:
            schema = self._infer_database_schema(side)
        elif conn_type in FILE_TYPES:
            schema = self._infer_file_schema(
                path=side["uploaded_path"],
                conn_type=conn_type,
                connection=connection,
            )
        elif conn_type in CLOUD_TYPES:
            schema = self._infer_file_schema(
                path=side["file_path"],
                conn_type=conn_type,
                connection=connection,
            )
        else:
            df = self.load_dataframe(side)
            schema = {field.name: field.dataType.simpleString() for field in df.schema.fields}
        with self._cache_lock:
            self._schema_cache[cache_key] = (now, schema)
        return schema

    def _load_database(self, side: dict[str, Any]) -> DataFrame:
        connection = side["connection"]
        table_name = side["table_name"]
        where_condition = (side.get("where_condition") or "1=1").strip()
        if where_condition.lower().startswith("where "):
            where_condition = where_condition[6:].strip()
        conn_type = connection["type"]

        if conn_type == "DATABASE_DUCKDB":
            db_path = connection.get("hostname")
            if not db_path:
                raise ValueError("DuckDB path missing in hostname")
            query = f"SELECT * FROM {table_name} WHERE {where_condition}"
            return (
                self.spark.read.format("jdbc")
                .option("url", f"jdbc:duckdb:{db_path}")
                .option("query", query)
                .option("driver", "org.duckdb.DuckDBDriver")
                .load()
            )

        if conn_type == "DATABASE_DELTA":
            root_path = connection.get("hostname")
            if not root_path:
                raise ValueError("Delta root path missing in hostname")
            table_path = str(Path(root_path) / table_name)
            df = self.spark.read.format("delta").load(table_path)
            return df.filter(F.expr(where_condition))

        if conn_type == "DATABASE_SNOWFLAKE":
            self._ensure_snowflake_connector_available()
            sf_options = self._build_snowflake_options(connection, db_name=connection["database_name"], schema_name=connection["schema_name"])
            query = f"SELECT * FROM {table_name} WHERE {where_condition}"
            return self.spark.read.format("snowflake").options(**sf_options).option("query", query).load()

        if conn_type == "DATABASE_ORACLE":
            jdbc_url = f"jdbc:oracle:thin:@{connection['hostname']}:{connection['port']}/{connection['database_name']}"
            query = f"(SELECT * FROM {table_name} WHERE {where_condition}) source_subquery"
            return (
                self.spark.read.format("jdbc")
                .option("url", jdbc_url)
                .option("user", connection["username"])
                .option("password", connection["password"])
                .option("dbtable", query)
                .option("driver", "oracle.jdbc.OracleDriver")
                .load()
            )

        raise ValueError(f"Unsupported database type: {conn_type}")

    def _load_file(
        self,
        path: str,
        conn_type: str,
        connection: dict[str, Any],
        schema_def: dict[str, str] | None,
    ) -> DataFrame:
        if not path:
            raise ValueError("File path is required")
        if conn_type in FILE_TYPES and not Path(path).exists():
            raise ValueError(f"Local file not found: {path}")
        reader = self.spark.read
        has_header = bool(connection.get("header", True))
        trim_space = bool(connection.get("trim_space", True))
        enclosed_by = connection.get("field_enclosed_by")

        if conn_type.endswith("DELIMITED"):
            delimiter = connection.get("field_delim") or ","
            if delimiter == "\\t":
                delimiter = "\t"
            reader = reader.option("header", has_header).option("sep", delimiter)
            reader = reader.option("ignoreLeadingWhiteSpace", trim_space).option("ignoreTrailingWhiteSpace", trim_space)
            reader = (
                reader.option("multiLine", True)
                .option("mode", "PERMISSIVE")
                .option("enforceSchema", False)
                .option("encoding", "UTF-8")
                .option("emptyValue", "")
                .option("nullValue", "")
                .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
                .option("dateFormat", "yyyy-MM-dd")
            )
            if enclosed_by and enclosed_by != "None":
                reader = reader.option("quote", enclosed_by)
            if schema_def:
                df = reader.option("inferSchema", False).csv(path)
            else:
                df = reader.option("inferSchema", True).option("samplingRatio", 0.25).csv(path)
            return self._normalize_columns(df)

        if conn_type.endswith("EXCEL"):
            # Requires spark-excel package at runtime.
            reader = (
                reader.format("com.crealytics.spark.excel")
                .option("header", has_header)
                .option("ignoreLeadingWhiteSpace", trim_space)
                .option("ignoreTrailingWhiteSpace", trim_space)
                .option("inferSchema", True)
                .option("treatEmptyValuesAsNulls", True)
                .option("maxRowsInMemory", 10000)
            )
            df = reader.load(path)
            return self._normalize_columns(df)

        raise ValueError(f"Unsupported file type: {conn_type}")

    def _infer_database_schema(self, side: dict[str, Any]) -> dict[str, str]:
        connection = side["connection"]
        conn_type = connection["type"]
        table_name = side.get("table_name") or ""
        db_name, schema_name, pure_table_name = self._parse_table_parts(table_name, connection)

        if conn_type == "DATABASE_DUCKDB":
            db_path = connection.get("hostname")
            query = (
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                f"WHERE table_schema = '{schema_name}' AND table_name = '{pure_table_name}' "
                "ORDER BY ordinal_position"
            )
            df = (
                self.spark.read.format("jdbc")
                .option("url", f"jdbc:duckdb:{db_path}")
                .option("query", query)
                .option("driver", "org.duckdb.DuckDBDriver")
                .load()
            )
            return {row["column_name"]: row["data_type"] for row in df.collect()}

        if conn_type == "DATABASE_SNOWFLAKE":
            self._ensure_snowflake_connector_available()
            sf_options = self._build_snowflake_options(
                connection,
                db_name=db_name or connection["database_name"],
                schema_name=schema_name or connection["schema_name"],
            )
            query = (
                "SELECT COLUMN_NAME AS column_name, DATA_TYPE AS data_type "
                "FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_CATALOG = '{(db_name or connection['database_name']).upper()}' "
                f"AND TABLE_SCHEMA = '{(schema_name or connection['schema_name']).upper()}' "
                f"AND TABLE_NAME = '{pure_table_name.upper()}' "
                "ORDER BY ORDINAL_POSITION"
            )
            df = self.spark.read.format("snowflake").options(**sf_options).option("query", query).load()
            return {row["column_name"]: row["data_type"] for row in df.collect()}

        if conn_type == "DATABASE_ORACLE":
            owner = (schema_name or connection.get("schema_name") or "").upper()
            jdbc_url = f"jdbc:oracle:thin:@{connection['hostname']}:{connection['port']}/{connection['database_name']}"
            query = (
                "(SELECT column_name, data_type FROM all_tab_columns "
                f"WHERE owner = '{owner}' AND table_name = '{pure_table_name.upper()}' "
                "ORDER BY column_id) source_subquery"
            )
            df = (
                self.spark.read.format("jdbc")
                .option("url", jdbc_url)
                .option("user", connection["username"])
                .option("password", connection["password"])
                .option("dbtable", query)
                .option("driver", "oracle.jdbc.OracleDriver")
                .load()
            )
            return {row["column_name"]: row["data_type"] for row in df.collect()}

        if conn_type == "DATABASE_DELTA":
            root_path = connection.get("hostname")
            table_path = str(Path(root_path) / table_name)
            # Delta schema is available in metadata; this avoids full-data scan.
            df = self.spark.read.format("delta").load(table_path)
            return {field.name: field.dataType.simpleString() for field in df.schema.fields}

        raise ValueError(f"Unsupported database type: {conn_type}")

    def _infer_file_schema(self, path: str, conn_type: str, connection: dict[str, Any]) -> dict[str, str]:
        has_header = bool(connection.get("header", True))
        trim_space = bool(connection.get("trim_space", True))
        enclosed_by = connection.get("field_enclosed_by")

        if conn_type.endswith("DELIMITED"):
            delimiter = connection.get("field_delim") or ","
            if delimiter == "\\t":
                delimiter = "\t"
            sampling_ratio = self._sampling_ratio_for_path(path)
            reader = (
                self.spark.read.option("header", has_header)
                .option("sep", delimiter)
                .option("ignoreLeadingWhiteSpace", trim_space)
                .option("ignoreTrailingWhiteSpace", trim_space)
                # Sampling ratio prevents full file scanning for inference.
                .option("inferSchema", True)
                .option("samplingRatio", sampling_ratio)
            )
            if enclosed_by and enclosed_by != "None":
                reader = reader.option("quote", enclosed_by)
            df = reader.csv(path)
            df = self._normalize_columns(df)
            return {field.name: field.dataType.simpleString() for field in df.schema.fields}

        if conn_type.endswith("EXCEL"):
            reader = self.spark.read.format("com.crealytics.spark.excel").option("header", has_header)
            reader = reader.option("ignoreLeadingWhiteSpace", trim_space).option("ignoreTrailingWhiteSpace", trim_space)
            reader = reader.option("inferSchema", True).option("treatEmptyValuesAsNulls", True)
            df = self._normalize_columns(reader.load(path).limit(1000))
            return {field.name: field.dataType.simpleString() for field in df.schema.fields}

        raise ValueError(f"Unsupported file type: {conn_type}")

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

    def _sampling_ratio_for_path(self, path: str) -> float:
        try:
            size_bytes = Path(path).stat().st_size
        except OSError:
            return self.sample_ratio_medium

        if size_bytes < 50 * 1024 * 1024:
            return self.sample_ratio_small
        if size_bytes < 500 * 1024 * 1024:
            return self.sample_ratio_medium
        return self.sample_ratio_large

    def _build_snowflake_options(self, connection: dict[str, Any], db_name: str, schema_name: str) -> dict[str, str]:
        """Build Snowflake connector options for both auth modes."""
        sf_options = {
            "sfURL": connection["hostname"],
            "sfUser": connection["username"],
            "sfDatabase": db_name,
            "sfSchema": schema_name,
        }
        role_name = connection.get("role_name")
        if role_name:
            sf_options["sfRole"] = role_name

        auth_type = (connection.get("authtype") or "").lower()
        if auth_type == "externalbrowser":
            sf_options["sfAuthenticator"] = "externalbrowser"
        else:
            password = connection.get("password")
            if password:
                sf_options["sfPassword"] = password
        return sf_options

    def _ensure_snowflake_connector_available(self) -> None:
        """Best-effort Snowflake connector class probe."""
        candidates = [
            "net.snowflake.spark.snowflake.DefaultSource",
            "net.snowflake.spark.snowflake.Utils",
            "net.snowflake.spark.snowflake.SnowflakeRelationProvider",
            "net.snowflake.spark.snowflake.SnowflakeSource",
        ]
        try:
            for class_name in candidates:
                try:
                    self.spark._jvm.java.lang.Class.forName(class_name)
                    return
                except Exception:
                    continue
        except Exception:
            return
        # Do not hard-fail here; let actual Snowflake read emit the connector's
        # native error for compatibility/auth/network issues.
        return

    def _apply_schema(self, df: DataFrame, schema_def: dict[str, str]) -> DataFrame:
        for col_name, col_type in schema_def.items():
            if col_name in df.columns:
                df = df.withColumn(col_name, F.col(col_name).cast(col_type))
        return df

    def _normalize_columns(self, df: DataFrame) -> DataFrame:
        new_names: list[str] = []
        used: dict[str, int] = {}
        for idx, original in enumerate(df.columns):
            base = (str(original).replace("\ufeff", "").strip() or f"COL_{idx + 1}")
            if base in used:
                used[base] += 1
                name = f"{base}_{used[base]}"
            else:
                used[base] = 1
                name = base
            new_names.append(name)

        for old, new in zip(df.columns, new_names):
            if old != new:
                df = df.withColumnRenamed(old, new)
        return df
