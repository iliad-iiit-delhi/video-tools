import kivy
from kivy.app import App

from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import  FloatLayout
from kivy.config import Config
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from functools import wraps
from kivy.clock import Clock

import mido
import cv2
import numpy as np
import time

Config.set('kivy', 'keyboard_mode', 'systemandmulti')

rate_transfer = 0
message_count = 0
rtmidi = None
midiports = []
outputport = None
vidcap = None
success = False
stopped = True
# def send_RGB(r):

# 	# Just sending the MIDO messages
# 	msg = mido.Message('control_change', channel=1-1, control=16, value=r)
# 	outport.send(msg)
# 	print('Sent message :', msg)
# 	# outport.send(mido.Message('control_change', channel=1+1, control=17, value=g))
# 	# outport.send(mido.Message('control_change', channel=1+1, control=17, value=b))


# def OnSliderValueChange(instance,value):
# 	run_dict["v"] = value
# 	print(value, run_dict['v'])

# decorator function
def yield_to_sleep(func):
	@wraps(func) # takes a function used in a decorator and adds the functionality of copying over the function name, arguments list, etc. 
	def wrapper(*args, **kwargs):
		gen = func()
		def next_step(*_): #defining a custom iterator function
			try:
				t = next(gen)  # this executes the part of 'func' (in this case, read_image) before the yield statement and returns control to you
				# i.e in this case it returns run_dict["v"] and stores it in t
			except StopIteration:
				pass
			else:
				start = time.time()
				Clock.schedule_once(next_step, t)  # having control you can resume func execution after some time
				print("Process time: " + str(time.time() - start))
				# this executes the part of func after the yield
		next_step()
	return wrapper

# generator function
@yield_to_sleep  # use this decorator to cast 'yield' to non-blocking sleep
def read_video():
	global rate_transfer, success, stopped
	# for i in range(10):
	while(success and not stopped):
		# yield run_dict["v"]  # use yield to "sleep"
		yield rate_transfer
		# value of run_dict["v"] is returned and the generator state is suspended
		# During the next call the generator resumes where it freezed before and then the value of r is randomly generated. 
		# r = random.randint(10, 90)
		rgb = send_message_midi()
		# g = random.randint(, 40)
		# b = random.randint(30, 50)
		# send_RGB(r)


def send_message_midi():
	
	global vidcap, message_count, outputport, success
	# cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
	# print(dt)
	success,image = vidcap.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# print('Read a new frame with shape: ', gray.shape)
	message_count+=1

	# for i in range(5):
	note = np.mean(gray)
	# print(note)
	# note = np.mean(note)
	# print(note)
	note=note//2
	# print(note)
	# print(type(note))
	# msg = mido.Message('note_on', note=int(note))
	msg = mido.Message('control_change', channel=1-1, control=16, value=int(note))
	outputport.send(msg)
	print('Sent message ', msg, " : " , message_count)
	# time.sleep(sleep_time)


class kivi_app(App):

	def OnRunButtonPressed(self, instance):
		global vidcap, success, stopped
		stopped = False
		success = True
		vidcap = cv2.VideoCapture('test_video.mp4')
		print(vidcap)
		# success,image = self.vidcap.read()
		read_video()
		# self.event.cancel()
		# self.event = Clock.schedule_interval(self.send_next_message_callback,self.rate_transfer)
		# return event
			

	def OnStopButtonPressed(self, instance):
		# pass
		global stopped
		stopped = True
		# if self.event != None:
		# 	Clock.unschedule(self.event)

	def OnSliderValueChange(self, instance,value):
		global rate_transfer
		self.title_label.text = "Rate of transfer (delay) in secs: " + str(value)
		rate_transfer = value
		print(value, rate_transfer)
		# self.event = Clock.schedule_interval(self.send_next_message_callback,self.rate_transfer)

		#TODO: interrupt the timer of the sending of messages

	def build(self):

		FLOAT_LAYOUT = FloatLayout(size=(300, 300))

		self.title_label = Label(text="Rate of transfer (delay) in secs: 1",
						  font_size=20,
						  pos_hint={'x': .4, 'y': .8},
						  size_hint=(.2, .2))

		self.text_box = TextInput(multiline=False,
							 font_size=20,
							 pos_hint={'x': .4, 'y': .3},
							 size_hint=(.2, .2)
							 )

		self.run_button = Button(text='Run',
							font_size=20,
							pos_hint={'x': .3, 'y': .1},
							size_hint=(.2, .1),
							# on_press=OnRunButtonPressed
							)

		self.stop_button = Button(text='Stop',
							font_size=20,
							pos_hint={'x': .6, 'y': .1},
							size_hint=(.2, .1),
							# on_press=OnRunButtonPressed
							)


		self.slider1 = Slider(min=1,
					 max=10, 
					 value=1,
					 step = 1,
					 pos_hint={'x': .1, 'y': .1},
					 size_hint=(.2, .1),
					 )


		FLOAT_LAYOUT.add_widget(self.title_label)
		# FLOAT_LAYOUT.add_widget(text_box)
		FLOAT_LAYOUT.add_widget(self.run_button)
		FLOAT_LAYOUT.add_widget(self.stop_button)
		FLOAT_LAYOUT.add_widget(self.slider1)

		global rate_transfer, message_count, rtmidi, midiports, outputport

		rate_transfer = self.slider1.value
		message_count = 0
		rtmidi = mido.Backend('mido.backends.rtmidi')
		# portmidi = mido.Backend('mido.backends.portmidi')

		# inputport = rtmidi.open_input()
		# outputport = rtmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
		# outputport = portmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
		try: 
			midiports = mido.get_output_names()
			# print(midiports[1])
			outputport = rtmidi.open_output(midiports[1]) #try 

			print(cv2.__version__)
			# self.event = None
			self.slider1.bind(value = self.OnSliderValueChange)
			self.run_button.bind(on_press= self.OnRunButtonPressed)
			self.stop_button.bind(on_press = self.OnStopButtonPressed)
		except Exception as e:
			self.title_label.text = str(e)
			print(str(e))
		
		return FLOAT_LAYOUT

	def calculate(self, *args):
		print(args)


if __name__ == '__main__':

	kivi_app().run()

	# TODO: need to put the following stuff on a different thread