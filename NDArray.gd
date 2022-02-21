class_name NDArray
extends Resource

var rows:int = 0
var columns:int = 0
var data:Array[float] = []
var broadcast:bool = false  # If true, will wrap all reads.

func _init(num_rows:int, num_columns:int, init_value = null):
	self.rows = num_rows
	self.columns = num_columns
	self.data.resize(self.rows*self.columns)
	if init_value != null:
		if init_value is float:
			self.fill(init_value)
		elif init_value is Array:
			self.data = init_value

func fill(value):
	for i in range(len(self.data)):
		self.data[i] = value

func to_string():
	var result = "[\n"
	for i in range(self.rows):
		result += "["
		for j in range(self.columns):
			result += str(self.get_value(i, j))
			result += ", "
		result += "], \n"
	result += "]"
	return result

func set_value(i:int, j:int, value:float):
	# TODO: Assert i and j in range?
	assert(i >= 0 and i < self.rows and j >= 0 and j <= self.columns)
	self.data[j + i*self.columns] = value

func get_value(i:int, j:int) -> float:
	if not self.broadcast:
		assert(i >= 0 and i < self.rows and j >= 0 and j <= self.columns)
	else:
		i = i % self.rows
		j = j % self.columns
	return self.data[j + i*self.columns]

func set_row(i:int, row:Array):
	assert(len(row) == self.columns)
	for j in range(self.columns):
		self.set_value(i, j, row[j])

func set_column(j:int, column:Array):
	assert(len(column) == self.rows)
	for i in range(self.rows):
		self.set_value(i, j, column[j])

func foreach_unary(op):
	var result = get_script().new(self.rows, self.columns)
	
	for x in range(len(self.data)):
		# op.call is GDScript 4.0
		result.data[x] = op.call(self.data[x])
		#result.data[x] = op.call_func(self.data[x])
	
	return result

func foreach_binary(op, other):
	var result = get_script().new(self.rows, self.columns)

	# If our sizes match and we are not broadcasting, can co-iterate:
	if self.rows == other.rows and self.columns == other.columns:
		# Can use the fast co-iteration trick:
		for x in range(len(self.data)):
			result.data[x] = op.call(self.data[x], other.data[x])
			#result.data[x] = op.call_func(self.data[x], other.data[x])
	else:
		# This is slower, but will handle broadcasting:
		for i in range(self.rows):
			for j in range(self.columns):
				result.set_value(i, j, op.call(self.get_value(i, j), other.get_value(i, j)))
				#result.set_value(i, j, op.call_func(self.get_value(i, j), other.get_value(i, j)))
	
	return result

func _maybe_broadcast(value):
	if value is int or value is float:
		var temp = value
		value = get_script().new(1, 1)
		value.set_value(0, 0, temp)
		value.broadcast = true
	return value

func _addition_internal(a, b):
	return a+b

func _subtraction_internal(a, b):
	return a-b

func _multiplication_internal(a, b):
	return a*b

func _division_internal(a, b):
	return a/b

func _negate_internal(a):
	return -a

func _leaky_relu_internal(a):
	return max(0.01*a, a)

func _abs_internal(a):
	return abs(a)

func add(value):
	value = self._maybe_broadcast(value)
	# We don't have curried functions in GDScript or lambdas.  :'(
	var op = Callable(self, "_addition_internal")  # GDScript 4.0
	#var op = funcref(self, "_addition_internal")
	return self.foreach_binary(op, value)

func subtract(value):
	value = self._maybe_broadcast(value)
	var op = Callable(self, "_subtraction_internal")
	return self.foreach_binary(op, value)

func multiply(value):
	# Dot product.
	value = self._maybe_broadcast(value)
	var op = Callable(self, "_multiplication_internal")
	return self.foreach_binary(op, value)

func divide(value):
	value = self._maybe_broadcast(value)
	var op = Callable(self, "_division_internal")
	return self.foreach_binary(op, value)

func negate():
	var op = Callable(self, "_negate_internal")
	return self.foreach_unary(op)

func leaky_relu():
	var op = Callable(self, "_leaky_relu_internal")
	return self.foreach_unary(op)

func abs():
	var op = Callable(self, "_abs_internal")
	return self.foreach_unary(op)

func sum():
	var accumulator:float = 0.0
	for i in range(len(self.data)):
		accumulator += self.data[i]
	return accumulator

func matmul(other):
	assert(self.columns == other.rows)
	var result = get_script().new(self.rows, other.columns)
	for i in range(self.rows):
		for j in range(other.columns):
			var accumulator = 0
			for k in range(self.columns):
				accumulator += self.get_value(i, k)*other.get_value(k, j)
			result.set_value(i, j, accumulator)
	return result

func transpose():
	var result = get_script().new(self.columns, self.rows)
	for i in range(self.rows):
		for j in range(self.columns):
			result.set_value(j, i, self.get_value(i, j))
	return result
