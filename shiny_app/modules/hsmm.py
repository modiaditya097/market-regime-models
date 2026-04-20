"""HSMM Model Tab."""

from shiny import module, ui, render
import pandas as pd
from pathlib import Path
import base64

@module.ui
def model_tab_ui(cfg: dict):
    return ui.page_fluid(
        ui.div(
            ui.h2("HSMM: 5-State Hidden Semi-Markov Model", class_="text-primary"),
            ui.p("S&P 500 Regime Detection with Duration Modeling", class_="lead"),
            ui.hr(),
            
            ui.layout_columns(
                ui.card(
                    ui.card_header("Model Overview"),
                    ui.markdown("""
                    **Approach:** Hidden Semi-Markov Model with 5 states
                    
                    **Features:** Returns, volatility, volume, momentum + duration
                    
                    **States:** 5 regimes with explicit duration modeling
                    
                    **Parameters:** ~60 estimated parameters
                    
                    **Training Period:** 2013-2022
                    
                    **Test Period:** 2023-2024
                    """)
                ),
                ui.card(
                    ui.card_header("Performance Metrics"),
                    ui.output_table("metrics")
                ),
                col_widths=[6, 6]
            ),
            
            ui.hr(),
            
            ui.card(
                ui.card_header("Cumulative Returns Comparison"),
                ui.output_ui("cumulative_plot")
            ),
            
            ui.hr(),
            
            ui.card(
                ui.card_header("Regime Classification Over Time"),
                ui.output_ui("timeline_plot")
            ),
            
            ui.hr(),
            
            ui.card(
                ui.card_header("Regime Characteristics"),
                ui.output_ui("characteristics_plot")
            ),
            
            ui.hr(),
            
            ui.card(
                ui.card_header("Transition Matrix"),
                ui.output_ui("transition_plot")
            ),
        )
    )

@module.server
def model_tab_server(input, output, session, cfg: dict, project_root):
    output_dir = Path(cfg["output_dir"])
    
    @render.table
    def metrics():
        results_file = output_dir / "results.csv"
        if not results_file.exists():
            return pd.DataFrame({"Status": ["No data"]})
        
        df = pd.read_csv(results_file)
        display = []
        display.append({"Metric": "Sharpe Ratio", "Value": f"{df['sharpe'].iloc[0]:.3f}"})
        display.append({"Metric": "Max Drawdown", "Value": f"{df['max_drawdown'].iloc[0]:.2f}%"})
        display.append({"Metric": "Volatility", "Value": f"{df['volatility'].iloc[0]:.2f}%"})
        display.append({"Metric": "Active Return", "Value": f"{df['active_return'].iloc[0]:.2f}%"})
        display.append({"Metric": "Turnover", "Value": f"{df['turnover'].iloc[0]:.4f}"})
        return pd.DataFrame(display)
    
    @render.ui
    def cumulative_plot():
        plot_file = output_dir / "plots/cumulative_returns.png"
        if not plot_file.exists():
            return ui.p("Plot not found", class_="text-muted")
        encoded = base64.b64encode(plot_file.read_bytes()).decode()
        return ui.img(
            src=f"data:image/png;base64,{encoded}",
            style="width:100%;max-width:1000px;display:block;margin:auto"
        )
    
    @render.ui
    def timeline_plot():
        plot_file = output_dir / "plots/regime_timeline.png"
        if not plot_file.exists():
            return ui.p("Plot not found", class_="text-muted")
        encoded = base64.b64encode(plot_file.read_bytes()).decode()
        return ui.img(
            src=f"data:image/png;base64,{encoded}",
            style="width:100%;max-width:1000px;display:block;margin:auto"
        )
    
    @render.ui
    def characteristics_plot():
        plot_file = output_dir / "plots/regime_characteristics.png"
        if not plot_file.exists():
            return ui.p("Plot not found", class_="text-muted")
        encoded = base64.b64encode(plot_file.read_bytes()).decode()
        return ui.img(
            src=f"data:image/png;base64,{encoded}",
            style="width:100%;max-width:1000px;display:block;margin:auto"
        )
    
    @render.ui
    def transition_plot():
        plot_file = output_dir / "plots/transition_matrix.png"
        if not plot_file.exists():
            return ui.p("Plot not found", class_="text-muted")
        encoded = base64.b64encode(plot_file.read_bytes()).decode()
        return ui.img(
            src=f"data:image/png;base64,{encoded}",
            style="width:100%;max-width:1000px;display:block;margin:auto"
        )
