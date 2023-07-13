from graphing import Test

foc = "RTSA"
bg = "BG"

test = Test.read_test("logs_dress_all2.csv")
test.time_series_by_category(foc, bg)
