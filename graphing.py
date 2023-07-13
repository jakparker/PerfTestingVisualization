import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates


class UnitTest:
    """A processed set of results indicating the average response time and thread rate of a particular action."""

    def __init__(self, label, master_df, interval=60):
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
        # Rename the columns to 'average_elapsed' and 'value_count'
        resampled_df.rename(
            columns={"Latency": "avg_res", "threadName": "txn_per_sec"},
            inplace=True,
        )
        resampled_df["txn_per_sec"] = resampled_df["txn_per_sec"] / interval

        # Add the resampled dataframe to the interval_dataframes dictionary
        self.results = resampled_df

    @classmethod
    def label_is_in(self, ls):
        return lambda unit_test: unit_test.label in ls

    @classmethod
    def has_category(self, cat):
        return lambda unit_test: unit_test.category == cat

    def change_interval(self, new_int):
        self.interval_duration = pd.Timedelta(seconds=new_int) 
        self.results = self.df.resample(self.interval_duration).agg(
            {"Latency": np.mean, "threadName": "count"}
        )


class Test:
    def __init__(self, df, test_segment_intervals, metric_intervals):
        '''
        df: DataFrame to analyze
        test_segment_intervals: Length of each test run
        metric_intervals: interval to average metrics across
        '''
        self.results = df
        unit_tests = []

        self.unique_labels = df["label"].unique()

        for label in self.unique_labels:
            unit_tests.append(UnitTest(label, df, metric_intervals))

        self.unit_tests = unit_tests
        
        self.metric_intervals = metric_intervals
        self.test_segment_intervals = test_segment_intervals

    @classmethod
    def read_test(self, location, test_segment_intervals, metric_intervals):
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
    
    def update_metric_int(self, new_int):
        '''
        update the metric interval for all the unit tests in test
        '''
        self.unit_tests = [ut.change_interval(new_int) for ut in self.unit_tests]
        
    def gen_colour_dict(self):
        '''
        generate a dictoinary that maps a colour
        '''

    def time_series_by_labels(self, focus_labels, bg_labels, title, metric):
        return self.time_series(
            UnitTest.label_is_in(focus_labels), UnitTest.label_is_in(bg_labels),
            title, metric
        )

    def time_series_highlight_category(self, cat):
        # TODO: Show time series graph with a category highlighted, and the rest of the data as 'background'
        pass

    def time_series_by_category(self, focus_category,title):
        focus = [ut for ut in self.unit_tests if ut.has_category(focus_category)]
        bg = [ut for ut in self.unit_tests if not ut.has_category(focus_category)]
        return self.time_series(focus, bg)

    def time_series_unit(self, unit_test_label, title, metric='avg_res'):
        return self.time_series_by_labels([unit_test_label], [], title, metric)

    def time_series(self, focus_fn, bg_fn, title, metric='avg_res'):
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

        nlabels = len(focus_tests) + len(bg_tests) #store number of labels, used for legend formatting
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
        if metric == 'avg_res':
            ax.set_ylabel("Response Time (ms)")
        elif metric == 'txn_per_sec':
            ax.set_ylabel('Transactions Per Second')

        # Set the title
        ax.set_title(title)
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M')) #Format Timestamps on xaxis
        # Add a legend below the graph, 
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=(nlabels+1)//2)

        # Rotate the x-axis tick labels for better visibility (optional)
        plt.xticks(rotation=45)
        plt.grid()
        # Display the plot
        plt.show()
