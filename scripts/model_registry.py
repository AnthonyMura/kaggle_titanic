from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from IPython.display import display


@dataclass
class ModelRecord:
    model_name: str
    pipeline: Any
    best_parameters: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self) -> None:
        self._records: Dict[str, ModelRecord] = {}

    def register(
            self,
            model_name: str,
            pipeline: Any,
            best_parameters: Dict[str, Any],
            metrics: Optional[Dict[str, float]] = None,
            metadata: Optional[Dict[str, Any]] = None
            ) -> None:
        record = ModelRecord(
                model_name=model_name,
                pipeline=pipeline,
                best_parameters=best_parameters,
                metrics=metrics if metrics is not None else {},
                metadata=metadata if metadata is not None else {}
                )
        self._records[model_name] = record

    def get(self, model_name: str) -> ModelRecord:
        if model_name not in self._records:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        return self._records[model_name]

    def remove(self, model_name: str) -> None:
        if model_name not in self._records:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        del self._records[model_name]

    def reset(self) -> None:
        self._records.clear()

    def list_models(self) -> List[str]:
        return list(self._records.keys())

    def to_dataframe(self, sort_by: Optional[str] = None, ascending: bool = False) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for record in self._records.values():
            row: Dict[str, Any] = {
                "model_name"     : record.model_name,
                "best_parameters": str(record.best_parameters),
                }
            row.update(record.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        if sort_by is None:
            metric_columns = [col for col in df.columns if col not in {"model_name", "best_parameters"}]
            if metric_columns:
                sort_by = metric_columns[0]

        if sort_by:
            if sort_by not in df.columns:
                raise ValueError(f"Metric '{sort_by}' not found in registered models.")
            df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

        return df

    def show(self, sort_by: Optional[str] = None, ascending: bool = False) -> None:
        df = self.to_dataframe(sort_by=sort_by, ascending=ascending)
        if df.empty:
            print("No models registered.")
            return

        metric_columns = [col for col in df.columns if col not in {"model_name", "best_parameters"}]

        if metric_columns:
            format_mapping = {col: "{:.4f}" for col in metric_columns}
            styled_df = df.style.format(format_mapping, na_rep="-")
            if sort_by and sort_by in df.columns:
                styled_df = styled_df.highlight_max(subset=[sort_by], color="#d4edda")
            display(styled_df)
        else:
            display(df)