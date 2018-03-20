import timeit 

def test_time(stmt ,stepup, number = 10000):
    t = timeit.timeit(stmt = stmt, stepup = stepup, number=number)
    t = t/number
    print(stmt,'Average run time:',t)