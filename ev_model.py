import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go


DAYS_OF_THE_YEAR = 365
TRIPS_PER_DAY = 2
LIKELY_COMMUTING_HOURS = [9, 16]
COMMUTING_HOURS_PROB = 0.3
NON_COMMUTING_HOURS = 0.05


# making up a price forecast to inform the charge schedule
def price_forecast(plug_in_time, plug_out_time):
    prices = []
    times = []
    hours_of_day = plug_in_time

    while hours_of_day != plug_out_time:
        if 17 <= hours_of_day <= 22:
            # peak hours - highest prices
            price = random.gauss(0.30, 0.05)
        elif hours_of_day == 23 or 0 <= hours_of_day <= 6:
            # night hours - lowest prices
            price = random.gauss(0.10, 0.05)
        else:
            # off-peak daytime - medium prices
            price = random.gauss(0.20, 0.10)

        prices.append(price)
        times.append(hours_of_day)
        hours_of_day = (hours_of_day + 1) % 24

    return pd.Series(prices, index=times)


class EVModel:
    """
    The EV class simulates the charging and driving behaviour of an individual EV.
    """

    def __init__(self, archetype_df_row):
        # archetype data
        self.archetype_name = archetype_df_row["Name"]
        self.time_plugged_in = int(archetype_df_row["Plug-in time"])
        self.time_plugged_out = int(archetype_df_row["Plug-out time"])
        self.miles_per_day = (
            float(archetype_df_row["Miles/yr"]) / DAYS_OF_THE_YEAR
        )  # assuming an even distribution of miles per day
        self.avg_trip_length = (
            self.miles_per_day / TRIPS_PER_DAY
        )  # assuming two trips a day
        self.efficiency = float(archetype_df_row["Efficiency (mi/kWh)"])
        self.target_soc = float(archetype_df_row["Target SoC"])
        self.charger_kw = float(archetype_df_row["Charger kW"])
        self.plug_in_frequency = float(archetype_df_row["Plug-in frequency (per day)"])
        self.capacity_kwh = archetype_df_row["Battery (kWh)"]

        # initialize SoC at 00:00: SoC at plug-in time + random increase between 0-15% SoC as it has been plugged in for a few hours (cap at 100%)
        self.state_of_charge = min(
            100, float(archetype_df_row["Plug-in SoC"]) + random.uniform(0, 15)
        )

        # tracking variables
        self.plugged_in = False
        self.charging = False
        self.driving = False

        # setting the charge schedule
        self.charge_schedule = self.calc_charging_schedule()

    def update_plugin_status(self, hour_of_day):
        # once day check if EV will plug based on frequency
        if hour_of_day == self.time_plugged_in:
            if random.random() >= self.plug_in_frequency:
                self.plugged_in = False
                return

        # small chance deviating from normal schedule
        if random.random() < 0.05:
            # shifting the plug in/out times by +/- 1 hour and making sure they are within 0-23
            self.time_plugged_in = (self.time_plugged_in + random.choice([-1, 1])) % 24
            self.time_plugged_out = (
                self.time_plugged_out + random.choice([-1, 1])
            ) % 24

        # check if the EV is plugged in
        self.plugged_in = (hour_of_day >= self.time_plugged_in) or (
            hour_of_day <= self.time_plugged_out
        )

    def charge(self, kwh):
        soc_increase = (kwh / self.capacity_kwh) * 100
        # adding charge with a max of 100%
        self.state_of_charge = min(100, self.state_of_charge + soc_increase)

    def discharge(self, kwh):
        soc_decrease = (kwh / self.capacity_kwh) * 100
        # discharging with a min of 0%
        self.state_of_charge = max(0, self.state_of_charge - soc_decrease)

    def is_driving(self, hour_of_day):
        # get the probability and hourly miles for the given hour
        if hour_of_day in LIKELY_COMMUTING_HOURS:
            # higher prob at peak driving hours
            probs = COMMUTING_HOURS_PROB
        else:
            probs = NON_COMMUTING_HOURS

        # check if the EV is driving based on probs
        if random.random() <= probs:
            kwh_driven = self.avg_trip_length / self.efficiency
            self.discharge(kwh_driven)

    def calc_charging_schedule(self):
        # calculate charge schedule based on price forecast
        prices = price_forecast(self.time_plugged_in, self.time_plugged_out)
        # how many kWh needed to reach target SoC
        kwh_needed = (self.target_soc - self.state_of_charge) * self.capacity_kwh / 100
        # how many hours needed to charge
        hours_needed = int(np.ceil(kwh_needed / self.charger_kw))
        # sort for lowest prices
        return prices.index[np.argsort(prices)[:hours_needed]].tolist()

    def simulate_hour(self, hour_of_day):
        self.update_plugin_status(hour_of_day)
        # is it plugged in?
        if self.plugged_in:
            # charging?
            self.charging = hour_of_day in self.charge_schedule
            if self.charging:
                self.charge(self.charger_kw)
        else:
            self.is_driving(hour_of_day)

    def plot_daily_profile(self):
        """Plot daily profile for this EV"""

        hours = list(range(24))
        soc_values = []
        plugged_in_status = []

        # simulate 24 hours
        for hour in range(24):
            self.simulate_hour(hour)
            soc_values.append(self.state_of_charge)
            plugged_in_status.append(self.plugged_in)

        # plot
        plt.figure(figsize=(15, 8))
        plt.fill_between(
            hours,
            0,
            100,
            where=plugged_in_status,
            alpha=0.3,
            hatch="///",
            color="gray",
            label="Plugged In",
        )
        plt.plot(hours, soc_values, "r-", linewidth=2, label="Battery SoC")
        plt.xlabel("Hour of Day")
        plt.ylabel("Battery State of Charge (%)")
        plt.title(f"Daily Profile - {self.archetype_name}")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.xticks(range(24))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


class PopulationSimulator:
    """Simulates multiple EVs with different driving/charging patterns"""

    def __init__(self, archetype_df, total_population):
        # load archetype data
        self.archetype_df = archetype_df
        self.evs = []
        self.results = pd.DataFrame()

        # add EVs based on % of population for each archetype
        for _, archetype_row in self.archetype_df.iterrows():
            count = int(archetype_row["% of population"] * total_population / 100)

            for _ in range(count):
                ev = EVModel(archetype_row)
                self.evs.append(ev)

    def run_simulation(self, hours=24):
        # store results
        results = {
            "hour": [],
            "archetype": [],
            "soc": [],
            "plugged_in": [],
            "charging": [],
        }

        # run for each hour
        for hour in range(hours):
            for ev in self.evs:
                ev.simulate_hour(hour)
                results["hour"].append(hour)
                results["archetype"].append(ev.archetype_name)
                results["soc"].append(ev.state_of_charge)
                results["plugged_in"].append(ev.plugged_in)
                results["charging"].append(ev.charging)

        self.results = pd.DataFrame(results)
        return self.results

    def plot_population_results(self, percentile=95):
        # Create figure
        fig = go.Figure()

        # Create dropdown menu options
        dropdown_buttons = []
        # Add individual archetype options
        for arch in self.archetype_df["Name"]:
            dropdown_buttons.append(
                dict(
                    args=[
                        {
                            "visible": [
                                True if a == arch else False
                                for a in self.archetype_df["Name"]
                                for _ in range(3)
                            ]
                        }
                    ],
                    label=arch,
                    method="update",
                )
            )

        # Add traces for each archetype
        for arch in self.archetype_df["Name"]:
            # Filter results for this archetype
            results = self.results[self.results["archetype"] == arch]
            visible = True if arch == self.archetype_df["Name"].iloc[0] else False

            # Add mean SOC line
            soc_mean = results.groupby("hour")["soc"].mean()
            fig.add_trace(
                go.Scatter(
                    x=list(range(24)),
                    y=soc_mean.values,
                    name="Mean SOC",
                    line=dict(color="red", width=2),
                    visible=visible,
                )
            )

            # Add percentile lines
            soc_upper_percentile = results.groupby("hour")["soc"].quantile(
                percentile / 100
            )
            soc_lower_percentile = results.groupby("hour")["soc"].quantile(
                (100 - percentile) / 100
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(24)),
                    y=soc_upper_percentile.values,
                    name=f"{percentile}th Percentile",
                    line=dict(color="red", width=1.5, dash="dash"),
                    visible=visible,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(24)),
                    y=soc_lower_percentile.values,
                    name=f"{100-percentile}th Percentile",
                    line=dict(color="red", width=1.5, dash="dash"),
                    visible=visible,
                )
            )

        # Update layout
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="State of Charge (%)",
            title="Population-Level EV Charging Patterns",
            yaxis_range=[0, 100],
            xaxis_range=[-0.5, 23.5],
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(0,0,0,1)",
                font=dict(color="white"),
            ),
            height=800,
            width=1500,
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[
                                {
                                    "visible": [
                                        True if a == arch else False
                                        for a in self.archetype_df["Name"]
                                        for _ in range(3)
                                    ]
                                }
                            ],
                            label=arch,
                            method="update",
                        )
                        for arch in self.archetype_df["Name"]
                    ],
                    direction="down",
                    showactive=True,
                    x=0.83,
                    y=1.08,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,1)",  # Dropdown background
                    font=dict(color="black", size=12),  # Dropdown text color
                    bordercolor="rgba(255,255,255,0.3)",
                    pad={"r": 10, "t": 10},  # Padding
                )
            ],
        )

        return fig

    def plot_archetype_comparison_plotly(self, percentile=95):
        fig = go.Figure()
        colors = ["#FF0000", "#1DA1F2", "#9370DB", "purple", "orange", "cyan"]

        # Calculate average across all archetypes
        all_data = self.results.copy()
        soc_mean = all_data.groupby("hour")["soc"].mean()
        soc_upper = all_data.groupby("hour")["soc"].quantile(percentile / 100)
        soc_lower = all_data.groupby("hour")["soc"].quantile(0.05)

        # Add confidence interval for all archetypes
        fig.add_trace(
            go.Scatter(
                x=list(soc_mean.index) + list(soc_mean.index)[::-1],
                y=list(soc_upper.values) + list(soc_lower.values)[::-1],
                fill="toself",
                fillcolor=colors[0],
                line=dict(color="rgba(255,255,255,0)"),
                name="All Archetypes",
                opacity=0.2,
                legendgroup="all",
                showlegend=True,
            )
        )

        # Add mean line for all archetypes
        fig.add_trace(
            go.Scatter(
                x=soc_mean.index,
                y=soc_mean.values,
                line=dict(color=colors[0], width=2),
                name="All Archetypes",
                legendgroup="all",
                showlegend=False,
            )
        )

        # Update layout with hover and active styles for dropdown
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="State of Charge (%)",
            title="Population-Level EV Charging Patterns",
            yaxis_range=[0, 100],
            xaxis_range=[-0.5, 23.5],
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(0,0,0,1)",
                font=dict(color="white"),
            ),
            height=800,
            width=1500,
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[
                                {
                                    "visible": [
                                        True if a == arch else False
                                        for a in self.archetype_df["Name"]
                                        for _ in range(3)
                                    ]
                                }
                            ],
                            label=arch,
                            method="update",
                        )
                        for arch in self.archetype_df["Name"]
                    ],
                    direction="down",
                    showactive=True,
                    x=0.83,
                    y=1.08,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(50,50,50,1)",  # Dropdown background
                    font=dict(color="white"),  # Dropdown text color
                    bordercolor="rgba(255,255,255,0.3)",
                    pad={"r": 10, "t": 10},  # Padding
                    # Adjust hover and selected item styles
                    hoverlabel=dict(
                        bgcolor="rgba(100,100,100,1)",  # Hover background color
                        font=dict(color="white"),  # Font color on hover
                    ),
                    active=dict(
                        bgcolor="rgba(100,100,100,1)",  # Active background color
                        font=dict(color="white"),  # Font color when selected
                    ),
                )
            ],
        )

        return fig

    def _simulate_all_archetypes(self):
        """Runs one simulation for each archetype (for individual plot to select)"""
        results = {"hour": [], "archetype": [], "soc": [], "plugged_in": []}

        # one EV for each archetype
        for _, archetype_row in self.archetype_df.iterrows():
            ev = EVModel(archetype_row)

            # 24 hours
            for hour in range(24):
                ev.simulate_hour(hour)
                results["hour"].append(hour)
                results["archetype"].append(ev.archetype_name)
                results["soc"].append(ev.state_of_charge)
                results["plugged_in"].append(ev.plugged_in)

        return pd.DataFrame(results)

    def plot_individual_daily_profile(self):
        # simulate all archetypes
        df = self._simulate_all_archetypes()

        fig = go.Figure()

        # traces for each archetype (initially hidden)
        for archetype in df["archetype"].unique():
            archetype_data = df[df["archetype"] == archetype]

            # SOC line
            fig.add_trace(
                go.Scatter(
                    x=archetype_data["hour"],
                    y=archetype_data["soc"],
                    name=f"{archetype} - SOC",
                    line=dict(width=2),
                    visible=False,
                )
            )

            # plugged-in area as bar plot
            fig.add_trace(
                go.Bar(
                    x=archetype_data["hour"],
                    y=archetype_data["plugged_in"].map({True: 100, False: 0}),
                    name=f"Plugged In",
                    marker_color="rgba(128, 128, 128, 0.3)",
                    width=1,  # Make bars touch each other
                    visible=False,
                )
            )

        # first archetype visible by default
        fig.data[0].visible = True
        fig.data[1].visible = True

        # dropdown menu
        buttons = []
        for i, archetype in enumerate(df["archetype"].unique()):
            visible = [False] * len(fig.data)
            visible[i * 2] = True  # SOC line
            visible[i * 2 + 1] = True  # Plugged-in area
            buttons.append(
                dict(label=archetype, method="update", args=[{"visible": visible}])
            )

        # layout
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.90,  # Moved button more to the right
                    "y": 1.15,
                    "xanchor": "right",
                    "yanchor": "top",
                }
            ],
            xaxis_title="Hour of Day",
            yaxis_title="Battery State of Charge (%)",
            title="Daily EV Profile by Archetype",
            yaxis_range=[0, 100],
            xaxis_range=[-0.5, 23.5],
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
            height=800,
            width=1500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        )

        return fig
