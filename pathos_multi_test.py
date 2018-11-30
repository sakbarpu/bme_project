from pathos.multiprocessing import ProcessingPool as Pool

class Calculate:
	def run(self):
		def func(x):
			return x*x
		p = Pool()
		return p.map(func, [1,2,4])

c1 = Calculate()
print (c1.run())
