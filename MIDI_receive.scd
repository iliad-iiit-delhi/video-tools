MIDIClient.init;
MIDIIn.connectAll;
MIDIClient.sources;
MIDIClient.destinations;

(
m = MIDIFunc.noteOn({ |vel, num|
    "note % @ velocity %\n".postf(num, vel);
});
n = MIDIFunc.noteOff({ |vel, num|
    "note % @ velocity %\n".postf(num, vel);
});
)

m.free;
n.free;
//
//
// (
// var notes, on, off;
//
// notes = Array.newClear(128);    // array has one slot per possible MIDI note
//
// on = MIDIFunc.noteOn({ |veloc, num, chan, src|
// 	notes[num] = Synth(\default, [\freq, num.midicps,
// 	\amp, veloc * 0.00315]);
// });
//
// off = MIDIFunc.noteOff({ |veloc, num, chan, src|
// 	notes[num].release;
// });
//
// q = { on.free; off.free; };
// )
//
// // when done:
// q.value;