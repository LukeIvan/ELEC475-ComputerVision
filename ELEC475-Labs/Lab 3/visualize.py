import optuna
from plotly.io import show

if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna_study_new.db",
        direction="minimize",
        study_name="alexnet",
        load_if_exists=True,
    )
    fig = optuna.visualization.plot_slice(study)

    # Update marker styles for better visibility
    for trace in fig.data:
        trace.update(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="black")))

    # Update layout for better readability
    fig.update_layout(
        title="Optimization Slice Plot for Batch Size, Learning Rate, and Weight Decay",
        font=dict(size=14),
        title_x=0.5,
        plot_bgcolor="white",
        xaxis1_title="Batch Size",
        xaxis2_title="Learning Rate",
        xaxis3_title="Weight Decay",
        yaxis1_title="Best Validation Loss",
        yaxis2_title="Best Validation Loss",
        yaxis3_title="Best Validation Loss",

    )

    # Ensure gridlines, titles, and formatting are applied to all axes
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            title_text = fig.layout[axis].title.text.title() if fig.layout[axis].title and fig.layout[axis].title.text else None
            fig.layout[axis].update(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                title=title_text,
            )

    show(fig)
