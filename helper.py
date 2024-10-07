from math import ceil
import logging
import pretty_midi
import numpy as np
import random
import glob

velocity_step = 7
vocab_size = 128*2 + 100 + int(ceil(126/velocity_step))

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class MusicEvent:
    def __init__(self, start_time, event_type, value):
        self.start_time, self.event_type, self.value = start_time, event_type, value

    def encode(self):
        event_codes = {
            'note_on': lambda: self.value,
            'note_off': lambda: 128 + self.value,
            'time_shift': lambda: 128*2 + self.value,
            'velocity': lambda: 128*2 + 100 + self.value
        }
        return event_codes.get(self.event_type, lambda: None)()

    @staticmethod
    def decode(code):
        if code < 128:
            return 'note_on', code
        elif code < 128*2:
            return 'note_off', code - 128
        elif code < 128*2 + 100:
            return 'time_shift', (code - 128*2)/100 + 0.01
        else:
            return 'velocity', (code - 128*2 - 100)*velocity_step + int(velocity_step/2)

def midi_to_sequence(midi_input):
    midi_input = pretty_midi.PrettyMIDI(midi_input) if isinstance(midi_input, str) else midi_input
    instrument = midi_input.instruments[0]

    def create_events():
        current_velocity = 0
        for note in instrument.notes:
            if note.velocity != current_velocity:
                yield MusicEvent(note.start, 'velocity', int(min(note.velocity, 125)/velocity_step))
                current_velocity = note.velocity
            yield MusicEvent(note.start, 'note_on', note.pitch)
            yield MusicEvent(note.end, 'note_off', note.pitch)

    event_queue = sorted(create_events(), key=lambda x: x.start_time)

    def process_time_shifts():
        current_time = 0
        for event in event_queue:
            time_diff = event.start_time - current_time
            while time_diff > 0.01:
                time_step = min(time_diff, 1) - 0.01
                yield MusicEvent(current_time, 'time_shift', int(time_step*100))
                time_diff -= time_step
            current_time = event.start_time
            yield event

    full_event_queue = list(process_time_shifts())
    sequence = np.fromiter((event.encode() for event in full_event_queue), dtype=np.int32)

    assert np.max(sequence) < vocab_size
    return sequence

def sequence_to_midi(sequence):
    midi_output = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name='instrument')
    midi_output.instruments.append(instrument)

    if sequence.vocab_size > 1:
        sequence = np.argmax(sequence, axis=-1)

    active_notes, current_velocity, current_time = {}, 40, 0.

    def process_event(event):
        nonlocal current_velocity, current_time
        event_type, value = MusicEvent.decode(event)

        if event_type == 'time_shift':
            current_time += value
        elif event_type == 'velocity':
            current_velocity = value
            for note in active_notes.values():
                if note[2] == current_time:
                    note[0] = value
        elif event_type == 'note_on':
            if value in active_notes:
                logging.debug(f'consecutive note_on for pitch {value} at time {active_notes[value][2]} and {current_time}')
            else:
                active_notes[value] = [current_velocity, value, current_time, -1]
        elif event_type == 'note_off':
            if value in active_notes:
                note = active_notes[value]
                note[-1] = current_time
                if note[-1] > note[-2]:
                    instrument.notes.append(pretty_midi.Note(*note))
                else:
                    logging.debug(f'note with non-positive duration for pitch {note[1]} at time {note[2]}')
                del active_notes[value]
            else:
                logging.debug(f'note_off without note_on for pitch {value} at time {current_time}')

    for event in sequence:
        process_event(event)

    for note in active_notes.values():
        note[-1] = current_time
        if note[-1] > note[-2]:
            instrument.notes.append(pretty_midi.Note(*note))

    return midi_output

def segment_sequence(sequence, max_length=50):
    assert len(sequence) > max_length
    increment = int(max_length/2)

    segments = [np.concatenate(([128*2+1], sequence[:max_length]))]
    segments.extend(sequence[i:i+max_length+1] for i in range(increment, len(sequence)-max_length, increment))

    return np.stack(segments, axis=0)

def process_midi_sequences(all_midis=None, data_directory='data', num_segments=10000, max_length=50):
    if all_midis is None:
        all_midis = glob.glob('maestro-v1.0.0/**/*.midi')
        random.shuffle(all_midis)

    processed_data = []
    total_segments = 0

    for midi_file in all_midis:
        seq = segment_sequence(midi_to_sequence(midi_file), max_length)
        processed_data.append(seq)
        total_segments += len(seq)
        if total_segments > num_segments:
            break

    return np.vstack(processed_data)

def generate_random_midi(num_notes=100):
    midi_output = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name='instrument')
    midi_output.instruments.append(instrument)

    rng = np.random.default_rng()
    pitches = rng.integers(0, 128, size=num_notes)
    velocities = rng.integers(10, 80, size=num_notes)
    note_durations = np.abs(rng.standard_normal(num_notes) + 1)
    intervals = np.abs(0.2 * rng.standard_normal(num_notes) + 0.3)

    start_time = 0.5
    for i in range(num_notes):
        instrument.notes.append(pretty_midi.Note(velocities[i], pitches[i], start_time, start_time + note_durations[i]))
        start_time += intervals[i]

    return midi_output