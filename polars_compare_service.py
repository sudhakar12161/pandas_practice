from __future__ import annotations

import logging
import time
from typing import Any

import polars as pl

from app.services.compare_service import CompareService


class PolarsCompareService(CompareService):
    """Polars implementation while reusing reporting/export helpers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("data_compare.compare.polars")

    def resolve_schema(self, side: dict[str, Any], saved_schema: dict[str, Any] | None) -> dict[str, str]:
        start = time.perf_counter()
        schema_name = side.get("schema_name")
        if schema_name and schema_name not in {"NONE", "INFER_SCHEMA"} and saved_schema:
            schema = self._to_schema_map(saved_schema["schema_def"])
            self.logger.info("Polars resolve schema from saved columns=%s elapsed_ms=%.1f", len(schema), (time.perf_counter() - start) * 1000)
            return schema
        if schema_name == "INFER_SCHEMA":
            schema = self._to_schema_map(self.loader.infer_schema(side))
            self.logger.info("Polars resolve schema inferred columns=%s elapsed_ms=%.1f", len(schema), (time.perf_counter() - start) * 1000)
            return schema
        df = self.loader.load_dataframe(side)
        schema = {name: str(dtype) for name, dtype in df.schema.items()}
        self.logger.info("Polars resolve schema from dataframe columns=%s elapsed_ms=%.1f", len(schema), (time.perf_counter() - start) * 1000)
        return schema

    def analyze(self, side: dict[str, Any], exclude_columns: list[str], pk_columns: list[str]) -> dict[str, Any]:
        start = time.perf_counter()
        df = self.loader.load_dataframe(side)
        columns = [col for col in df.columns if col not in exclude_columns]
        total_rows = df.height
        metric_rows: list[dict[str, Any]] = []

        for metric_name in ["NOT_NULL_COUNT", "NULL_COUNT", "UNIQUE_COUNT", "DUPLICATE_COUNT", "MIN", "MAX"]:
            metric_rows.append({"metric": metric_name, "values": {}})

        def set_metric(metric_name: str, col_name: str, value: Any) -> None:
            row = next(item for item in metric_rows if item["metric"] == metric_name)
            row["values"][col_name] = value

        for col_name in columns:
            series = df.get_column(col_name)
            null_count = int(series.null_count())
            not_null_count = int(total_rows - null_count)
            unique_count = int(series.n_unique())
            duplicate_count = int(total_rows - unique_count)
            min_val = series.drop_nulls().min()
            max_val = series.drop_nulls().max()

            set_metric("NOT_NULL_COUNT", col_name, not_null_count)
            set_metric("NULL_COUNT", col_name, null_count)
            set_metric("UNIQUE_COUNT", col_name, unique_count)
            set_metric("DUPLICATE_COUNT", col_name, duplicate_count)
            set_metric("MIN", col_name, str(min_val))
            set_metric("MAX", col_name, str(max_val))

        result = {
            "dataset_name": self.get_dataset_name(side),
            "columns": columns,
            "rows": metric_rows,
            "pk_columns": pk_columns,
        }
        self.logger.info(
            "Polars analyze completed dataset=%s columns=%s rows=%s elapsed_ms=%.1f",
            result["dataset_name"],
            len(columns),
            total_rows,
            (time.perf_counter() - start) * 1000,
        )
        return result

    def compare_data(
        self,
        source_side: dict[str, Any],
        target_side: dict[str, Any],
        source_pk_columns: list[str],
        target_pk_columns: list[str],
        column_mapping: dict[str, str],
    ) -> dict[str, Any]:
        start = time.perf_counter()
        source_df = self.loader.load_dataframe(source_side)
        target_df = self.loader.load_dataframe(target_side)

        if len(source_pk_columns) != len(target_pk_columns):
            raise ValueError("Primary key column counts must match between Source and Target")

        pk_pairs = []
        for src_pk in source_pk_columns:
            tgt_pk = column_mapping.get(src_pk)
            if not tgt_pk:
                raise ValueError(f"Missing mapping for source PK column: {src_pk}")
            pk_pairs.append((src_pk, tgt_pk))

        source_pk_cols = [src for src, _ in pk_pairs]
        target_pk_cols = [tgt for _, tgt in pk_pairs]

        if len(set(target_pk_cols)) != len(target_pk_cols):
            raise ValueError("Each source PK must map to a unique target PK")
        if set(target_pk_cols) != set(target_pk_columns):
            raise ValueError("Mapped target PKs must exactly match selected target PK columns")

        missing_source = [col for col in source_pk_cols if col not in source_df.columns]
        missing_target = [col for col in target_pk_cols if col not in target_df.columns]
        if missing_source or missing_target:
            raise ValueError(f"PK columns missing. source_missing={missing_source}, target_missing={missing_target}")

        needed_source_cols = sorted(set(source_pk_cols + [col for col in column_mapping if col in source_df.columns]))
        needed_target_cols = sorted(set(target_pk_cols + [col for col in column_mapping.values() if col in target_df.columns]))
        source_df = source_df.select([pl.col(c) for c in needed_source_cols])
        target_df = target_df.select([pl.col(c) for c in needed_target_cols])

        source_total = source_df.height
        target_total = target_df.height
        source_unique = source_df.unique(subset=source_pk_cols).height
        target_unique = target_df.unique(subset=target_pk_cols).height

        source_work = source_df.with_columns(
            [
                pl.col(src).cast(pl.Utf8).str.strip_chars().alias(f"__pk_{idx}")
                for idx, src in enumerate(source_pk_cols)
            ]
        )
        target_pref = target_df.rename({col: f"t__{col}" for col in target_df.columns})
        target_work = target_pref.with_columns(
            [
                pl.col(f"t__{tgt}").cast(pl.Utf8).str.strip_chars().alias(f"__pk_{idx}")
                for idx, tgt in enumerate(target_pk_cols)
            ]
        )
        join_keys = [f"__pk_{idx}" for idx in range(len(source_pk_cols))]

        joined = source_work.join(target_work, on=join_keys, how="inner")
        matched_rows = joined.height

        source_keys = source_work.select(join_keys).unique()
        target_keys = target_work.select(join_keys).unique()
        rows_only_in_source = source_keys.join(target_keys, on=join_keys, how="anti").height
        rows_only_in_target = target_keys.join(source_keys, on=join_keys, how="anti").height
        not_matched_rows = rows_only_in_source + rows_only_in_target

        column_stats = {}
        mismatched_columns: set[str] = set()
        mismatch_expr = pl.lit(False)
        for source_col, target_col in column_mapping.items():
            if source_col in source_pk_cols:
                continue
            target_pref_col = f"t__{target_col}"
            if source_col not in source_work.columns or target_pref_col not in target_work.columns:
                continue
            compare_expr = self._norm_pl_expr(source_col) == self._norm_pl_expr(target_pref_col)
            matches = joined.filter(compare_expr).height
            mismatches = matched_rows - matches
            column_stats[source_col] = {
                "target_column": target_col,
                "matching_count": int(matches),
                "non_matching_count": int(mismatches),
            }
            if mismatches > 0:
                mismatched_columns.add(source_col)
            mismatch_expr = mismatch_expr | (~compare_expr)

        mismatch_count = joined.filter(mismatch_expr).height if matched_rows else 0
        data_payload = self._build_data_tab_payload_polars(
            source_work=source_work,
            target_work=target_work,
            source_pk_cols=source_pk_cols,
            target_pk_cols=target_pk_cols,
            column_mapping=column_mapping,
            join_keys=join_keys,
        )

        source_name = self.get_dataset_name(source_side)
        target_name = self.get_dataset_name(target_side)
        mismatch_columns_text = ", ".join(sorted(mismatched_columns)) if mismatched_columns else ""
        report = {
            "source": {
                "dataset_name": source_name,
                "total_rows": int(source_total),
                "unique_rows_by_pk": int(source_unique),
                "duplicate_rows_by_pk": int(source_total - source_unique),
                "rows_only_in_source": int(rows_only_in_source),
            },
            "target": {
                "dataset_name": target_name,
                "total_rows": int(target_total),
                "unique_rows_by_pk": int(target_unique),
                "duplicate_rows_by_pk": int(target_total - target_unique),
                "rows_only_in_target": int(rows_only_in_target),
            },
            "common": {
                "dataset_name": "COMMON",
                "matched_rows_by_pk": int(matched_rows),
                "not_matched_rows_by_pk": int(not_matched_rows),
                "mismatched_columns": mismatch_columns_text,
                "column_level_stats": column_stats,
                "mismatch_row_count": int(mismatch_count),
            },
        }
        report["excel_path"] = self._write_excel(report, data_payload, source_name)
        self.logger.info(
            "Polars compare completed source=%s target=%s matched=%s mismatched_rows=%s elapsed_ms=%.1f",
            source_name,
            target_name,
            matched_rows,
            mismatch_count,
            (time.perf_counter() - start) * 1000,
        )
        return report

    def _build_data_tab_payload_polars(
        self,
        source_work: pl.DataFrame,
        target_work: pl.DataFrame,
        source_pk_cols: list[str],
        target_pk_cols: list[str],
        column_mapping: dict[str, str],
        join_keys: list[str],
    ) -> dict[str, Any]:
        full_join = source_work.join(target_work, on=join_keys, how="full")

        headers: list[str] = []
        exprs: list[pl.Expr] = []
        pair_indices: list[tuple[int, int]] = []
        mismatch_expr = pl.lit(False)

        for src_pk, tgt_pk in zip(source_pk_cols, target_pk_cols):
            alias = f"PK_{src_pk}" if src_pk == tgt_pk else f"PK_{src_pk}_{tgt_pk}"
            headers.append(alias)
            exprs.append(pl.coalesce([pl.col(src_pk), pl.col(f"t__{tgt_pk}")]).alias(alias))

        for source_col, target_col in column_mapping.items():
            if source_col in source_pk_cols:
                continue
            tgt_col = f"t__{target_col}"
            if source_col not in full_join.columns or tgt_col not in full_join.columns:
                continue
            src_alias = f"SRC_{source_col}"
            tgt_alias = f"TGT_{target_col}"
            src_idx = len(headers)
            headers.extend([src_alias, tgt_alias])
            pair_indices.append((src_idx, src_idx + 1))
            exprs.extend([pl.col(source_col).alias(src_alias), pl.col(tgt_col).alias(tgt_alias)])
            mismatch_expr = mismatch_expr | (self._norm_pl_expr(source_col) != self._norm_pl_expr(tgt_col))

        source_present = pl.lit(True)
        for src_pk in source_pk_cols:
            source_present = source_present & pl.col(src_pk).is_not_null()
        target_present = pl.lit(True)
        for tgt_pk in target_pk_cols:
            target_present = target_present & pl.col(f"t__{tgt_pk}").is_not_null()

        status_col = (
            pl.when(source_present & (~target_present))
            .then(pl.lit("Only in Source"))
            .when((~source_present) & target_present)
            .then(pl.lit("Only in Target"))
            .when(source_present & target_present & mismatch_expr)
            .then(pl.lit("Mismatched"))
            .otherwise(pl.lit("Matched"))
            .alias("Matched_Ind")
        )
        headers.append("Matched_Ind")
        selected = full_join.select(exprs + [status_col]).head(self.max_error_rows)

        rows_payload: list[dict[str, Any]] = []
        for row in selected.iter_rows(named=True):
            values = [row.get(h) for h in headers]
            mismatch_indices: set[int] = set()
            if values[-1] == "Mismatched":
                for src_idx, tgt_idx in pair_indices:
                    if self._norm(values[src_idx]) != self._norm(values[tgt_idx]):
                        mismatch_indices.add(src_idx)
                        mismatch_indices.add(tgt_idx)
            rows_payload.append(
                {
                    "values": values,
                    "mismatch_indices": sorted(mismatch_indices),
                    "matched_ind": values[-1],
                }
            )
        return {"headers": headers, "rows": rows_payload}

    def _norm_pl_expr(self, col_name: str) -> pl.Expr:
        return pl.col(col_name).cast(pl.Utf8).str.strip_chars().fill_null("__NULL__")
