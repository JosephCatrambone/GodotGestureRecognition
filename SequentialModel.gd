class_name SequentialModel
extends Resource

var weights:Array[NDArray] = []
var biases:Array[NDArray] = []

func _init(json_model_data:String):
	var json = JSON.new()
	#var json_string = json.stringify(data_to_send)
	json.parse(json_model_data)
	var model_data = json.get_data()
	for idx in range(0, len(model_data['weights'])):  # Can't zip, so...
		var shape:Array[int] = model_data['shapes'][idx]
		var w:Array[float] = model_data['weights'][idx]
		var b:Array[float] = model_data['biases'][idx]
		var new_weight = NDArray.new(shape[0], shape[1])
		new_weight.data = w
		var new_bias = NDArray.new(1, shape[1])
		new_bias.data = b
		weights.append(new_weight)
		biases.append(new_bias)

func predict(data:Array[float]):
	var x = NDArray.new(1, len(data))
	x.data = data
	for idx in range(len(self.weights)):
		x = x.matmul(self.weights[idx])
		if idx != len(self.weights)-1:
			x = x.leaky_relu()
	return x
