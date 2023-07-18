import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates


class LineAppearance:
    def __init__(self, colour=None, marker="o"):
        """"""
        # Do we need other aspects here, or is this sufficient?
        # Will the marker only be present when the test is in the foreground?
        # TODO: replace this with a cycler/ iterator
        self.colour = colour
        self.marker = marker


class UnitTest:
    """A processed set of results indicating the average response time and thread rate of a particular action."""

    def __init__(self, label, master_df, interval=60, line_app=LineAppearance()):
        self.label = label
        self.category = label.split("_")[0]

        # Pull the relevant rows
        self.df = master_df[master_df["label"] == label].copy()

        # Define the interval duration in seconds
        self.interval_duration = pd.Timedelta(seconds=interval)

        # Convert the 'timeStamp' column to datetime if it's not already
        self.df["timeStamp"] = pd.to_datetime(self.df["timeStamp"], unit="ms")

        # Store the start and end times of the test before resampling
        self.start = self.df["timeStamp"].min()
        self.end = self.df["timeStamp"].max()

        # Set the 'timeStamp' column as the dataframe index
        self.df.set_index("timeStamp", inplace=True)

        # Resample the dataframe to 30-second intervals and calculate the average of 'elapsed' and the count of values
        resampled_df = self.df.resample(self.interval_duration).agg(
            {"Latency": np.mean, "threadName": "count"}
        )

        error_pct = (
            100
            - self.df["success"].astype(int).resample(self.interval_duration).mean()
            * 100
        )
        # Rename the columns to 'average_elapsed' and 'value_count'
        resampled_df.rename(
            columns={"Latency": "avg_res", "threadName": "txn_per_sec"},
            inplace=True,
        )
        resampled_df["txn_per_sec"] = resampled_df["txn_per_sec"] / interval
        resampled_df["error_pct"] = error_pct
        # Add the resampled dataframe to the interval_dataframes dictionary
        self.results = resampled_df

    @classmethod
    def label_is_in(self, ls):
        return lambda unit_test: unit_test.label in ls

    @classmethod
    def has_category(self, cat):
        return lambda unit_test: unit_test.category == cat

    @classmethod
    def not_has_category(self, cat):
        return lambda unit_test: unit_test.category != cat

    def change_interval(self, new_int):
        self.interval_duration = pd.Timedelta(seconds=new_int)
        self.results = self.df.resample(self.interval_duration).agg(
            {"Latency": np.mean, "threadName": "count"}
        )
        error_pct = (
            100
            - self.df["success"].astype(int).resample(self.interval_duration).mean()
            * 100
        )
        self.results.rename(
            columns={"Latency": "avg_res", "threadName": "txn_per_sec"},
            inplace=True,
        )
        self.results["txn_per_sec"] = self.results["txn_per_sec"] / new_int
        self.results["error_pct"] = error_pct


class Test:
    def __init__(self, df, metric_intervals, segmented=False):
        """
        df: DataFrame to analyze
        test_segment_intervals: Length of each test run
        metric_intervals: interval to average metrics across
        """
        self.results = df
        unit_tests = []

        self.unique_labels = df["label"].unique()

        for label in self.unique_labels:
            unit_tests.append(UnitTest(label, df, metric_intervals))

        self.unit_tests = unit_tests
        self.metric_intervals = metric_intervals

        if not segmented:
            # Segment the DF into each run, separated by Token Requests
            # Kinda of Janky way to do this. Want to generalize TODO
            self.results["Segment"] = -1

            indices = self.results[self.results["label"] == "Token_RTSA"].index

            segment_num = 0
            for i in indices:
                # Get all the rows between indicies, set them to segment num
                self.results.loc[
                    i : indices[indices > i].min(), "segment"
                ] = segment_num
                segment_num += 1

            # Set Tokens Back to -1
            self.results.loc[
                self.results["label"].str.startswith("Token_"), "segment"
            ] = -1

    @classmethod
    def read_test(self, location, metric_intervals):
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
        return Test(df, metric_intervals)

    def update_metric_int(self, interval):
        """
        update the metric interval for all the unit tests in test
        """
        self.unit_tests = [ut.change_interval(interval) for ut in self.unit_tests]

    def gen_colour_dict(self):
        """
        generate a dictoinary that maps a label (key) to a colour (value)
        To keep consistent colour  for each label over graphs
        """
        # TODO #2
        """
        class LineStyle
            colour
            marker
           
        {label: linestyle} 
        """

        pass

    def get_segments(self, segments):
        """
        Create new Test of only particular segments
        """
        new_df = self.results[self.results["segment"].isin(segments)]
        return Test(new_df, self.metric_intervals, segmented=True)

    def time_series_by_labels(self, focus_labels, bg_labels, title, metric):
        """
        Plot a foreground set of labels, and a background set of labels.
        """
        return self.time_series(
            UnitTest.label_is_in(focus_labels),
            UnitTest.label_is_in(bg_labels),
            title,
            metric,
        )

    def time_series_by_categories(self, focus_cat, bg_cat, title, metric):
        """
        Plot a foreground category and background category.
        """
        return self.time_series(
            UnitTest.has_category(focus_cat),
            UnitTest.has_category(bg_cat),
            title,
            metric,
        )

    def time_series_highlight_category(self, cat, title, metric):
        return self.time_series(
            UnitTest.has_category(cat), UnitTest.not_has_category(cat), title, metric
        )

    def time_series_by_category(self, focus_category, title, metric="avg_res"):
        focus = [ut for ut in self.unit_tests if ut.has_category(focus_category)]
        bg = [ut for ut in self.unit_tests if not ut.has_category(focus_category)]
        return self.time_series(focus, bg, title, metric)

    def time_series_unit(self, unit_test_label, title, metric="avg_res"):
        return self.time_series_by_labels([unit_test_label], [], title, metric)

    def time_series_dual(self, metric1, metric2):
        # TODO : Have two metric on the time series graph # 3

        pass

    def ninety_percentile(self, metric):
        pass

    def time_series(self, focus_fn, bg_fn, title, metric="avg_res"):
        """
        Create a time series graph

        X axis is confined by the 'focus labels',

        We require that focus_fn and bg_fn don't overlap.

        focus_fn and bg_fn are the criteria for being a focus or a background test.
        """
        plt.rcParams["figure.figsize"] = [30, 10]
        plt.rcParams.update({"font.size": 22})
        fig, ax = plt.subplots()

        # Store each of the focus and background tests
        focus_tests = [ut for ut in self.unit_tests if focus_fn(ut)]
        bg_tests = [ut for ut in self.unit_tests if bg_fn(ut)]

        nlabels = len(focus_tests) + len(
            bg_tests
        )  # store number of labels, used for legend formatting
        # Plot a line for each label
        for unit_test in focus_tests:
            # Convert the index (timeStamp) to a column for plotting
            dataframe = unit_test.results.reset_index()
            # Plot the line for the label
            ax.plot(
                dataframe["timeStamp"],
                dataframe[metric],
                label=unit_test.label,
                linewidth=3,
                marker="o",
                markersize=10,
                color=None,
            )

        for unit_test in bg_tests:
            dataframe = unit_test.results.reset_index()
            # Plot the line for the label
            ax.plot(
                dataframe["timeStamp"],
                dataframe[metric],
                label=unit_test.label,
                ls="--",
                alpha=0.5,
                color=None,
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
        if metric == "avg_res":
            ax.set_ylabel("Response Time (ms)")
        elif metric == "txn_per_sec":
            ax.set_ylabel("Transactions Per Second")

        # Set the title
        ax.set_title(title)
        ax.xaxis.set_major_formatter(
            dates.DateFormatter("%H:%M")
        )  # Format Timestamps on xaxis
        # Add a legend below the graph
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=(nlabels + 1) // 2
        )

        # Rotate the x-axis tick labels for better visibility (optional)
        plt.xticks(rotation=45)
        plt.grid()
        # Display the plot
        plt.show()
