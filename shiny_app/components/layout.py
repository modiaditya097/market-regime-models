from shiny import ui


def placeholder_card(message: str, run_btn_id: str | None = None) -> ui.Tag:
    """Centered card shown when a model's outputs do not exist yet."""
    btn = (
        ui.input_action_button(run_btn_id, "▶ Run Model", class_="btn-primary mt-3")
        if run_btn_id
        else ui.p("No run command configured for this model.", class_="text-muted mt-2")
    )
    return ui.div(
        ui.div(
            ui.h4("Results not yet available"),
            ui.p(message, class_="text-muted"),
            btn,
            class_="card-body text-center py-5",
        ),
        class_="card my-4",
        style="max-width:600px;margin:auto",
    )


def section(title: str, anchor_id: str, *content: ui.Tag) -> ui.Tag:
    """Labelled scrollable section with an anchor link target."""
    return ui.div(
        ui.h4(title, id=anchor_id, style="padding-top:1rem;border-bottom:1px solid #dee2e6;padding-bottom:.5rem"),
        *content,
        style="margin-bottom:2rem",
    )
