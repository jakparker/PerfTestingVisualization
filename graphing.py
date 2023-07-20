import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates


class LineAppearance:
    def __init__(self, color=None, marker="o", linestyle="-"):
        """"""
        # Do we need other aspects here, or is this sufficient?
        # Will the marker only be present when the test is in the foreground?
        # TODO: replace this with a cycler/ iterator
        self.color = color
        self.marker = marker
        self.linestyle = linestyle

    # def __copy__(self):
    #     copy_instance = LineAppearance()
    #     copy_instance.color = self.color
    #     copy_instance.marker = self.marker
    #     copy_instance.linestyle = self.linestyle
    #     return copy_instance


class UnitTest:
    """A processed set of results indicating the average response time and thread rate of a particular action."""

    def __init__(self, label, master_df, interval=60, line_app=LineAppearance()):
        self.label = label
        label_parts = label.split("_")
        self.category = label_parts[0]
        self.base = label_parts[-1] == "Base"
        self.line_app = line_app
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
    def category_is_in(self, ls):
        return lambda ut: ut.category in ls

    @classmethod
    def category_is_not_in(self, ls):
        return lambda ut: ut.category in ls

    @classmethod
    def base_and_category_is_in(self, ls):
        return lambda ut: ut.category in ls and ut.base

    @classmethod
    def not_base_and_category_is_in(self, ls):
        return lambda ut: ut.category in ls and not ut.base

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
        unit_tests_dict = {}
        unit_tests = unit_tests_dict.values()

        self.unique_labels = df["label"].unique()

        for label in self.unique_labels:
            unit_tests_dict.update({label: UnitTest(label, df, metric_intervals)})

        self.unit_tests = unit_tests
        self.unit_tests_dict = unit_tests_dict
        self.metric_intervals = metric_intervals

        segment_num = 0
        if not segmented:
            # Segment the DF into each run, separated by Token Requests
            # Kinda of Janky way to do this. Want to generalize TODO
            self.results["Segment"] = -1

            indices = self.results[self.results["label"] == "Token_Request"].index

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

        self.segment_num = segment_num
        self.segmented = segmented

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

    def gen_color_dict(self):
        """
        generate a dictoinary that maps a label (key) to a colour (value)
        To keep consistent colour  for each label over graphs
        """
        # TODO #2
        """
        class LineStyle
            color
            marker
           
        {label: linestyle} 
        """

        pass

    def get_avg_res_by_label(self, focus_label, bg_label):
        segment_avg_res_times = []
        for i in range(self.segment_num):
            segment = self.get_segments([i])
            if (
                focus_label in segment.unique_labels
                and bg_label in segment.unique_labels
            ):
                avg_res = segment.get_summary_metrics().at[focus_label, "mean"]
                segment_avg_res_times.append(avg_res)

        return np.mean(segment_avg_res_times)

    def get_summary_metrics(self):
        """Get count, mean, quantiles and other data. Only works correctly for a segment (the total duration is wrong)"""
        if self.segmented:
            total_seconds = (
                self.results["timeStamp"].max() - self.results["timeStamp"].min()
            ) / 1000
            summary_metrics = self.results.groupby("label")["Latency"].agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                    lambda x: x.quantile(0.9),
                    lambda x: x.quantile(0.95),
                    lambda x: x.quantile(0.99),
                    lambda x: x.shape[0] / total_seconds,
                ]
            )
            return summary_metrics
        else:
            # A quick fix; not ideal.
            return pd.DataFrame({"Error": ["not_segmented"]})

    def get_summary_metrics_by_segment(self):
        summary_metrics_list = []
        for i in range(self.segment_num):
            print("Segment:", i)
            segment = self.get_segments([i])
            # Append the results to the list
            summary_metrics_list.append(segment.get_summary_metrics())
        summary_metrics = pd.concat(
            summary_metrics_list, keys=range(11), names=["Segment"]
        )
        return summary_metrics

    def export_summary_metrics_by_segment(self, path="seg_stats.csv"):
        summary_metrics = self.get_summary_metrics_by_segment()
        summary_metrics.to_csv(path)

    def reset_line_apps(self):
        for ut in self.unit_tests:
            ut.line_app = LineAppearance()

    def assign_color_by_labels(self, label_color_dict):
        for label, color in label_color_dict.items():
            self.unit_tests_dict[label].line_app.color = color

    def assign_color_by_category(self, cat_color_dict):
        for cat, color in cat_color_dict.items():
            for ut in self.unit_tests:
                if ut.category == cat:
                    # print(ut.label, "| ", ut.category, "|", cat, " |", color)
                    ut.line_app.color = color

    def assign_line_app_by_labels(self, label_line_app_dict):
        for label, line_app in label_line_app_dict.items():
            self.unit_tests_dict[label].line_app = line_app.copy()

    def get_segments(self, segments):
        """
        Create new Test of only particular segments
        """
        new_df = self.results[self.results["segment"].isin(segments)]
        return Test(new_df, self.metric_intervals, segmented=True)

    def get_categories(self):
        categories = set([])
        for ut in self.unit_tests:
            categories.add(ut.category)
        return categories

    def time_series_by_labels(
        self, focus_labels, bg_labels, title="", metric="avg_res"
    ):
        """
        Plot a foreground set of labels, and a background set of labels.
        """
        return self.time_series(
            UnitTest.label_is_in(focus_labels),
            UnitTest.label_is_in(bg_labels),
            title,
            metric,
        )

    def time_series_by_label(self, foc_label, bg_label, title="", metric="avg_res"):
        return self.time_series_by_labels(
            [foc_label],
            [bg_label],
            title,
            metric,
        )

    def time_series_by_categories(
        self, focus_cats, bg_cats, title="", metric="avg_res"
    ):
        """
        Plot a set of foreground categories and background categories.
        """
        return self.time_series(
            UnitTest.category_is_in(focus_cats),
            UnitTest.category_is_in(bg_cats),
            title,
            metric,
        )

    def time_series_by_category(self, focus_cat, bg_cat, title="", metric="avg_res"):
        """Plot a single focus category against a single background category."""
        return self.time_series_by_categories([focus_cat], [bg_cat])

    def time_series_highlight_category(self, cat, title="", metric="avg_res"):
        """Highlight one category, and place on a background of the remaining categories."""
        return self.time_series(
            UnitTest.has_category(cat), UnitTest.not_has_category(cat), title, metric
        )

    def time_series_categories_against_base(self, cats, title="", metric="avg_res"):
        """Plot a set of categories against the base(s) in those categories."""
        return self.time_series(
            UnitTest.not_base_and_category_is_in(cats),
            UnitTest.base_and_category_is_in(cats),
        )

    def time_series_unit(self, unit_test_label, title="", metric="avg_res"):
        return self.time_series_by_labels([unit_test_label], [], title, metric)

    def time_series_dual(self, metric1, metric2):
        # TODO : Have two metric on the time series graph # 3

        pass

    def ninety_percentile(self, metric):
        pass

    def time_series(self, focus_fn, bg_fn, title="", metric="avg_res"):
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
                linewidth=5,
                marker=unit_test.line_app.marker,
                markersize=1,
                color=unit_test.line_app.color,
                ls=unit_test.line_app.linestyle,
            )

        for unit_test in bg_tests:
            dataframe = unit_test.results.reset_index()
            # Plot the line for the label
            ax.plot(
                dataframe["timeStamp"],
                dataframe[metric],
                label=unit_test.label,
                ls="--",
                alpha=1,
                color=unit_test.line_app.color,
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
        plt.show
