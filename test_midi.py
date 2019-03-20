import mido
import cv2
import numpy as np
import time

rtmidi = mido.Backend('mido.backends.rtmidi')
# portmidi = mido.Backend('mido.backends.portmidi')

# inputport = rtmidi.open_input()
# outputport = rtmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
# outputport = portmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
midiports = mido.get_output_names()
print(midiports[1])
outputport = rtmidi.open_output(midiports[1]) #try 

print(cv2.__version__)
vidcap = cv2.VideoCapture('test_video.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
	# cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
	success,image = vidcap.read()
	print(image.shape)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# print('Read a new frame with shape: ', gray.shape)
	count += 1

# for i in range(5):
	# note = np.mean(gray)
	r = image[:,:,0]
	g = image[:,:,1]
	b = image[:,:,2]
	note_r = np.mean(r)
	note_g = np.mean(g)
	note_b = np.mean(b)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	note = np.mean(gray)
	note=note//2
	# print(note_r, note_g, note_b)
	# break
	# note = np.mean(note)
	# print(note)
	# note=note//2
	# print(note)
	# print(type(note))
	msg1 = mido.Message('control_change', control= 0, channel = 0, value = int(note_r%127))
	msg2 = mido.Message('control_change', control=1, channel = 0, value = int(note_g%127))
	msg3 = mido.Message('control_change', control=2, channel = 0, value = int(note_b%127))
	outputport.send(msg1)
	outputport.send(msg2)
	outputport.send(msg3)
	
	print(count)
	print('Sent message ', msg1)
	print('Sent message ', msg2)
	print('Sent message ', msg3)

	time.sleep(0.3)

	# if count > 1000:
		# break

	# break
# for i in range(5):
# 	msg = mido.Message('note_off', note=50+i)
# 	outputport.send(msg)
# 	print('Sent message ', i)
