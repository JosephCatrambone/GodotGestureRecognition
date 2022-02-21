extends Control

var draw_started:bool = false
var gesture:Array[Vector2i] = []
var gesture_min_max:Array[Vector2i] = [Vector2i(), Vector2i()]
var model

var gesture_archetypes = {
	"spiral": NDArray.new(1, 10, [0.0609, 0.2293, -0.2562, 0.3954, -0.0708, 0.0164, -0.2815, -0.2233, -0.0494, 0.0157]),
	"star": NDArray.new(1, 10, [0.0039, 0.0173, 0.0178, -0.0094, 0.0641, 0.0473, 0.0622, -0.0149, 0.1001, -0.0475]),
	"wave": NDArray.new(1, 10, [-0.0871, -0.1207, -0.1085, -0.1800, -0.0187, -0.1468, -0.0587, 0.0519, -0.1696, -0.1321]),
}

func _ready():
	var file = File.new()
	file.open("res://result_model.json", File.READ)
	model = SequentialModel.new(file.get_as_text())

func _start_gesture(mouse_xy: Vector2i):
	self.draw_started = true
	gesture = [mouse_xy]
	gesture_min_max = [Vector2i(mouse_xy), Vector2i(mouse_xy)]

func _update_gesture(mouse_xy: Vector2i, mouse_dxdy: Vector2i):
	# New min:
	gesture_min_max[0] = Vector2i(mini(mouse_xy.x, gesture_min_max[0].x), mini(mouse_xy.y, gesture_min_max[0].y))
	# New max:
	gesture_min_max[1] = Vector2i(maxi(mouse_xy.x, gesture_min_max[1].x), maxi(mouse_xy.y, gesture_min_max[1].y))
	# Add point to list.
	gesture.append(mouse_xy)

func _complete_gesture():
	var min_xy = Vector2(gesture_min_max[0])
	var wh = Vector2(gesture_min_max[1])-min_xy
	if len(gesture) > 2 and wh.x > 1 and wh.y > 0:
		# Convert our strokes into a picture, flatten the picture, and pass it through the network.
		var image:Array[float] = []
		var image_width = 32
		var image_height = 32
		# Zero fill our image.
		for idx in range(image_width*image_height):
			image.append(0.0)
		# Rasterize the strokes in a naive way.
		for idx in range(len(self.gesture)-1):
			var p1 = Vector2(self.gesture[idx])
			var p2 = Vector2(self.gesture[idx+1])
			var dxdy = p2-p1
			var length = 1+(dxdy.length()/5)
			for step in range(0, int(length)):
				# Until this point, xy is still in the global stroke scale.  
				var xy = (dxdy*(float(step)/length) + p1) - min_xy;  # Zero offset.
				var x = xy.x/(wh.x+1e-6)
				var y = xy.y/(wh.y+1e-6)
				image[int(x*image_width) + int(y*image_height)*image_width] = 1.0
		# Now we have a 32x32 image in a flat buffer that we can match to a gesture.
		var gesture_name = self.match_gesture(image)
		print(gesture_name)
			
	self.gesture = []
	self.gesture_min_max = [Vector2i(), Vector2i()]
	self.draw_started = false

func distances_to_archetypes(flat_image: Array[float]):
	# Make an embedding and find the distance to each archetype.
	var pred = self.model.predict(flat_image)
	var distances := {}
	for name in self.gesture_archetypes:
		var vec = self.gesture_archetypes[name]
		# We can use abs diff here OR cosine similarity and do 1-cosine_sim.
		#var diff = pred.subtract(vec).abs().sum()
		var diff = pred.multiply(vec).sum()
		distances[name] = 1.0-diff
	return distances

func match_gesture(flat_image: Array[float]):
	var distances = self.distances_to_archetypes(flat_image)
	# We use these to select the closest match.
	var min_distance = 1000
	var min_name = ""
	for name in distances:
		var d = distances[name]
		# Pick best match.
		if d < min_distance:
			min_distance = d
			min_name = name
	return min_name

func _unhandled_input(event):
	if event is InputEventMouseMotion:
		var xy = event.position
		var dxdy = event.relative
		_update_gesture(xy, dxdy)
	elif event is InputEventMouseButton:
		if (not self.draw_started) and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
			_start_gesture(event.position)
		elif self.draw_started and (not event.pressed) and event.button_index == MOUSE_BUTTON_LEFT:
			_complete_gesture()
