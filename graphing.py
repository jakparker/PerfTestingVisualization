import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class UnitTest:
    """A processed set of results indicating the average response time and thread rate of a particular action."""

    def __init__(self, label, master_df):
        self.label = label
        self.category = label.split("_")[0]

        # Pull the relevant rows
        df = master_df[master_df["label"] == label].copy()

        # Define the interval duration in seconds (30 seconds in this case)
        interval_duration = pd.Timedelta(seconds=30)

        # Convert the 'timeStamp' column to datetime if it's not already
        df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="ms")

        # Store the start and end times of the test before resampling
        self.start = df["timeStamp"].min()
        self.end = df["timeStamp"].max()

        # Set the 'timeStamp' column as the dataframe index
        df.set_index("timeStamp", inplace=True)

        # Resample the dataframe to 30-second intervals and calculate the average of 'elapsed' and the count of values
        resampled_df = df.resample(interval_duration).agg(
            {"Latency": np.mean, "threadName": "count"}
        )
        # Rename the columns to 'average_elapsed' and 'value_count'
        resampled_df.rename(
            columns={"Latency": "avg_res", "threadName": "txn_per_sec"},
            inplace=True,
        )
        resampled_df["txn_per_sec"] = resampled_df["txn_per_sec"] / 30

        # Add the resampled dataframe to the interval_dataframes dictionary
        self.results = resampled_df

    @classmethod
    def label_is_in(self, ls):
        return lambda unit_test: unit_test.label in ls

    @classmethod
    def has_category(self, cat):
        return lambda unit_test: unit_test.category == cat


class Test:
    def __init__(self, df):
        self.results = df
        unit_tests = []

        self.unique_labels = df["label"].unique()

        for label in self.unique_labels:
            unit_tests.append(UnitTest(label, df))

        self.unit_tests = unit_tests

    @classmethod
    def read_test(self, location):
        imported = pd.read_csv(location)
        df = imported[
            [
                "timeStamp",
                "elapsed",
                "label",
                "threadName",
                "success",
                "responseCode",
                "grpThreads",
                "allThreads",
                "Latency",
                "IdleTime",
                "Connect",
            ]
        ]
        return Test(df)

    def time_series_by_labels(self, focus_labels, bg_labels):
        return self.time_series(
            UnitTest.label_is_in(focus_labels), UnitTest.label_is_in(bg_labels)
        )

    # def time_series_by_category(self, focus_category, bg_category):
    #     return self.time_series(
    #         UnitTest.has_category(focus_category), UnitTest.has_category(bg_category)
    #     )

    def time_series_by_category(self, focus_category):
        focus = [ut for ut in self.unit_tests if ut.has_category(focus_category)]
        bg = [ut for ut in self.unit_tests if not ut.has_category(focus_category)]
        return self.time_series(focus, bg)
  
    def time_series_unit(self, unit_test_label):
        return self.time_series_by_labels([unit_test_label], [])

    def time_series(self, focus_fn, bg_fn):
        """
        Create a time series graph

        X axis is confined by the 'focus labels',

        We require that focus_fn and bg_fn don't overlap.

        focus_fn and bg_fn are the criteria for being a focus or a background test.
        """
        plt.rcParams["figure.figsize"] = [30, 10]

        fig, ax = plt.subplots()

        # Store each of the focus and background tests
        focus_tests = [ut for ut in self.unit_tests if focus_fn(ut)]
        bg_tests = [ut for ut in self.unit_tests if bg_fn(ut)]

        # Plot a line for each label
        for unit_test in focus_tests:
            # Convert the index (timeStamp) to a column for plotting
            dataframe = unit_test.results.reset_index()
            # Plot the line for the label
            ax.plot(
                dataframe["timeStamp"],
                dataframe["avg_res"],
                label=unit_test.label,
                linewidth=3,
            )

        for unit_test in bg_tests:
            dataframe = unit_test.results.reset_index()
            # Plot the line for the label
            ax.plot(
                dataframe["timeStamp"],
                dataframe["avg_res"],
                label=unit_test.label,
                ls="--",
                alpha=0.5,
            )

        focus_start_times = [ut.start for ut in focus_tests]
        focus_end_times = [ut.end for ut in focus_tests]
        try:
            min_focus = pd.to_datetime(
                np.min(focus_start_times),
                unit="ms",
            )
            max_focus = pd.to_datetime(
                np.max(focus_end_times),
                unit="ms",
            )
            # Set the x-axis label and format
            ax.set_xlim(min_focus, max_focus)
        except ValueError:
            pass
        # Set the y-axis label
        ax.set_ylabel("Response Time")

        # Set the title
        ax.set_title("Graph")

        # Add a legend
        ax.legend()

        # Rotate the x-axis tick labels for better visibility (optional)
        plt.xticks(rotation=45)
        plt.grid()
        # Display the plot
        plt.show()
