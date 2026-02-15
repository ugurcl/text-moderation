from prometheus_client import Counter, Histogram

PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions",
    ["label", "allowed"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

FEEDBACK_COUNT = Counter(
    "feedback_total",
    "Total number of feedback submissions",
    ["predicted_label", "correct_label"],
)
