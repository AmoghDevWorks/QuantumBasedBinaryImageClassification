"""Streamlit UI for the quantum image classification demonstration."""

import io
import json

import pandas as pd
import streamlit as st

from src.classical_model import train_classical_model
from src.data_loader import load_binary_digits
from src.evaluation import evaluate_model
from src.preprocessing import prepare_data
from src.quantum_model import train_quantum_model
from src.utils import ExperimentConfig
from src.visualization import plot_accuracy_comparison, plot_confusion_matrices, plot_samples

st.set_page_config(page_title="Quantum Machine Learning for Image Classification", layout="wide")
st.markdown(
    """<style>
    .stApp { background: #07111f; color: #f5f7fb; }
    [data-testid="stSidebar"] { background: #0b1c31; }
    h1, h2, h3 { color: #53d8fb; }
    </style>""",
    unsafe_allow_html=True,
)

st.title("Quantum Machine Learning for Image Classification")
st.caption("Exploring Variational Quantum Classifiers Using Qiskit")
st.write(
    "This offline demonstration compares a simulated Variational Quantum Classifier "
    "(VQC) with a classical SVM on two handwritten digit classes. The quantum circuit "
    "runs on classical hardware through Qiskit Aer; no quantum advantage is implied."
)

with st.sidebar:
    st.header("Experiment configuration")
    class_a = st.number_input("Digit class A", 0, 9, 3)
    class_b = st.number_input("Digit class B", 0, 9, 7)
    n_features = st.selectbox("PCA features / qubits", [2, 4, 8], index=1)
    sample_limit = st.slider("Training sample limit", 20, 500, 200, step=20)
    test_size = st.slider("Test set size", 0.1, 0.4, 0.25, step=0.05)
    maxiter = st.slider("COBYLA iterations", 1, 100, 20)
    shots = st.select_slider("Simulator shots", options=[64, 128, 256, 512], value=256)
    seed = st.number_input("Random seed", 0, 100000, 42)
    train = st.button("Train models", type="primary", use_container_width=True)

config = ExperimentConfig(class_a, class_b, n_features, sample_limit, test_size, maxiter, shots, seed)
try:
    config.validate()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

dataset = load_binary_digits(
    config.class_a, config.class_b, config.test_size, config.seed, config.sample_limit
)
prepared = prepare_data(dataset, config.n_features, config.test_size, config.seed)

overview, explorer, predictions, circuit_tab, evaluation_tab = st.tabs(
    ["Overview", "Dataset Explorer", "Predictions", "Quantum Circuit", "Evaluation"]
)
with overview:
    st.subheader("Hybrid quantum-classical workflow")
    st.markdown("**Images → PCA fitted on training data → angle encoding → VQC/SVM → metrics**")
    st.info("Qubits, gates, superposition, and measurement are simulated locally by Qiskit Aer.")
    st.write("The digits dataset contains 8×8 grayscale images. PCA reduces the input to the selected circuit size.")

with explorer:
    st.subheader("Dataset explorer")
    st.pyplot(plot_samples(dataset.images, dataset.labels), clear_figure=True)
    st.write(f"Samples: {len(dataset.labels)} | Original image: 8×8 | PCA: {prepared.x_train.shape[1]} features")
    st.dataframe(pd.DataFrame(prepared.x_train[:8], columns=[f"component_{i+1}" for i in range(prepared.x_train.shape[1])]))

if train:
    with st.status("Training models...", expanded=True) as status:
        try:
            q_model, circuit, q_time = train_quantum_model(
                prepared.x_train,
                prepared.y_train,
                config.n_features,
                config.maxiter,
                config.shots,
                config.seed,
            )
            c_model, c_time = train_classical_model(prepared.x_train, prepared.y_train, config.seed)
            evaluations = [
                evaluate_model("Quantum VQC", q_model, prepared.x_test[:, : config.n_features], prepared.y_test, (config.class_a, config.class_b), q_time),
                evaluate_model("Classical SVM", c_model, prepared.x_test, prepared.y_test, (config.class_a, config.class_b), c_time),
            ]
            st.session_state["result"] = {
                "config": config,
                "quantum": q_model,
                "classical": c_model,
                "circuit": circuit,
                "evaluations": evaluations,
                "prepared": prepared,
            }
            status.update(label="Training complete", state="complete")
        except Exception as exc:
            status.update(label="Training failed", state="error")
            st.exception(exc)

result = st.session_state.get("result")
if result is not None and result["config"].as_dict() != config.as_dict():
    st.session_state.pop("result")
    result = None
with predictions:
    st.subheader("Predictions")
    if result is None:
        st.info("Train the models to enable predictions.")
    else:
        index = st.number_input("Test sample", 0, len(result["prepared"].y_test) - 1, 0)
        image = result["prepared"].test_images[index]
        st.image(image, width=160, clamp=True)
        row = {"Ground truth": int(result["prepared"].y_test[index])}
        for evaluation in result["evaluations"]:
            row[evaluation.model_name] = int(evaluation.predictions[index])
        st.dataframe(pd.DataFrame([row]))

with circuit_tab:
    st.subheader("Actual generated quantum circuit")
    if result is None:
        st.info("Train the models to construct the circuit.")
    else:
        circuit = result["circuit"]
        st.code(str(circuit.draw(output="text")))
        st.write({"qubits": circuit.num_qubits, "operations": sum(circuit.count_ops().values()), "depth": circuit.depth()})

with evaluation_tab:
    st.subheader("Evaluation and comparison")
    if result is None:
        st.info("Train the models to view real evaluation results.")
    else:
        rows = [dict(model=e.model_name, training_time_seconds=e.training_time, **e.metrics) for e in result["evaluations"]]
        metrics = pd.DataFrame(rows)
        st.dataframe(metrics, use_container_width=True)
        st.pyplot(plot_accuracy_comparison(result["evaluations"]), clear_figure=True)
        fig = plot_confusion_matrices(result["evaluations"])
        st.pyplot(fig, clear_figure=True)
        st.subheader("Classification reports")
        for evaluation in result["evaluations"]:
            report = pd.DataFrame(evaluation.report).transpose()
            st.markdown(f"**{evaluation.model_name}**")
            st.dataframe(report, use_container_width=True)
        st.download_button("Download metrics CSV", metrics.to_csv(index=False), "metrics.csv", "text/csv")
        predictions_df = pd.DataFrame({"actual": result["prepared"].y_test})
        for evaluation in result["evaluations"]:
            predictions_df[evaluation.model_name] = evaluation.predictions
        st.download_button("Download predictions CSV", predictions_df.to_csv(index=False), "predictions.csv", "text/csv")
        st.download_button("Download experiment JSON", json.dumps(config.as_dict(), indent=2), "experiment.json", "application/json")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        st.download_button("Download confusion matrices PNG", buffer.getvalue(), "confusion-matrices.png", "image/png")
        winner = max(result["evaluations"], key=lambda item: item.metrics["accuracy"])
        st.info(f"On this split, {winner.model_name} had the higher accuracy ({winner.metrics['accuracy']:.1%}).")
