import sounddevice as sd
import queue
import threading
import time
import numpy as np
import wave
import os
from collections import defaultdict
from datetime import datetime


class MultiMicSynchronizer:
    def __init__(self, device_list, samplerate=44100, blocksize=1024, save_to_file=True, output_dir="recordings"):
        self.device_list = device_list
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.sync_queues = defaultdict(queue.Queue)
        self.master_clock = None
        self.streams = []
        self.is_recording = False

        # File storage options
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.audio_buffers = defaultdict(list)  # Store all audio data for file writing
        self.recording_start_time = None

        # Create output directory if saving to file
        if self.save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)

    def synchronized_callback(self, device_id):
        """Create a callback function for a specific device"""
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio status for device {device_id}: {status}")

            # Use the first device as master clock
            current_time = time.inputBufferAdcTime
            if self.master_clock is None:
                self.master_clock = current_time

            # Calculate offset from master clock
            offset = current_time - self.master_clock

            # Store audio data
            audio_data = indata[:, 0].copy()  # Single channel per device

            # Store in queue for real-time access
            self.sync_queues[device_id].put({
                'data': audio_data,
                'timestamp': current_time,
                'offset': offset,
                'frames': frames,
                'device_id': device_id
            })

            # Store in buffer for file writing
            if self.save_to_file:
                self.audio_buffers[device_id].append(audio_data)

        return callback

    def start_recording(self):
        """Start recording from all devices simultaneously"""
        self.is_recording = True
        self.master_clock = None
        self.recording_start_time = datetime.now()

        # Clear buffers
        for device_id in self.device_list:
            self.audio_buffers[device_id].clear()
            # Clear queues
            while not self.sync_queues[device_id].empty():
                try:
                    self.sync_queues[device_id].get_nowait()
                except queue.Empty:
                    break

        try:
            # Create and start a stream for each device
            for device_id in self.device_list:
                stream = sd.InputStream(
                    device=device_id,
                    channels=1,  # Single channel per device
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    callback=self.synchronized_callback(device_id)
                )
                stream.start()
                self.streams.append(stream)
                print(f"Started recording from device {device_id}")

            print("All devices recording. Press Enter to stop...")
            if self.save_to_file:
                print(f"Audio will be saved to: {self.output_dir}/")
            input()

        except Exception as e:
            print(f"Error starting recording: {e}")
        finally:
            self.stop_recording()

    def stop_recording(self):
        """Stop all recording streams and save files if enabled"""
        self.is_recording = False

        # Stop streams
        for stream in self.streams:
            if stream.active:
                stream.stop()
                stream.close()
        self.streams.clear()

        # Save to files if enabled
        if self.save_to_file and self.recording_start_time:
            self.save_recordings()

            # Якщо записувалося 2 пристрої, об'єднати їх у стерео
            if len(self.device_list) == 2:
                timestamp = self.recording_start_time.strftime("%Y%m%d_%H%M%S")
                left_path = os.path.join(self.output_dir, f"mic_{self.device_list[0]}_{timestamp}.wav")
                right_path = os.path.join(self.output_dir, f"mic_{self.device_list[1]}_{timestamp}.wav")
                stereo_path = os.path.join(self.output_dir, f"stereo_{timestamp}.wav")

                try:
                    self.combine_files_to_stereo(left_path, right_path, stereo_path)
                except Exception as e:
                    print(f"Не вдалося об'єднати файли в стерео: {e}")

        print("Recording stopped")

    def save_recordings(self):
        """Save recorded audio to WAV files"""
        timestamp = self.recording_start_time.strftime("%Y%m%d_%H%M%S")

        for device_id in self.device_list:
            if self.audio_buffers[device_id]:
                # Concatenate all audio chunks
                full_audio = np.concatenate(self.audio_buffers[device_id])

                # Create filename
                filename = f"mic_{device_id}_{timestamp}.wav"
                filepath = os.path.join(self.output_dir, filename)

                # Save as WAV file
                self.save_wav_file(filepath, full_audio)
                print(f"Saved audio from device {device_id} to: {filepath}")

    def save_wav_file(self, filepath, audio_data):
        """Save audio data to a WAV file"""
        # Convert to 16-bit integers
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(self.samplerate)
            wav_file.writeframes(audio_int16.tobytes())

    def get_synchronized_data(self, timeout=1.0):
        """Get synchronized data from all devices (real-time access)"""
        synchronized_data = {}

        for device_id in self.device_list:
            try:
                data = self.sync_queues[device_id].get(timeout=timeout)
                synchronized_data[device_id] = data
            except queue.Empty:
                print(f"No data available from device {device_id}")

        return synchronized_data

    def get_recording_info(self):
        """Get information about current recording session"""
        info = {
            'devices': self.device_list,
            'samplerate': self.samplerate,
            'blocksize': self.blocksize,
            'is_recording': self.is_recording,
            'save_to_file': self.save_to_file,
            'output_dir': self.output_dir if self.save_to_file else None,
            'start_time': self.recording_start_time
        }

        if self.save_to_file:
            info['buffer_sizes'] = {
                device_id: len(self.audio_buffers[device_id])
                for device_id in self.device_list
            }

        return info
    
    def combine_files_to_stereo(self, filepath_left, filepath_right, output_filepath):
        """Об'єднати два монофоничні WAV-файли у стереофонічний WAV"""

        with wave.open(filepath_left, 'rb') as left_wav, wave.open(filepath_right, 'rb') as right_wav:
            # Перевірка параметрів файлів
            if left_wav.getnchannels() != 1 or right_wav.getnchannels() != 1:
                raise ValueError("Вхідні файли повинні бути монофонічними")
            if left_wav.getframerate() != right_wav.getframerate():
                raise ValueError("Вхідні файли повинні мати однакову частоту дискретизації")
            if left_wav.getsampwidth() != right_wav.getsampwidth():
                raise ValueError("Вхідні файли повинні мати однакову розрядність")

            n_frames = min(left_wav.getnframes(), right_wav.getnframes())
            framerate = left_wav.getframerate()
            sampwidth = left_wav.getsampwidth()

            # Зчитування даних
            left_data = left_wav.readframes(n_frames)
            right_data = right_wav.readframes(n_frames)

            # Конвертація байтів у numpy масив
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
            left_array = np.frombuffer(left_data, dtype=dtype)
            right_array = np.frombuffer(right_data, dtype=dtype)

            # Створення стерео масиву: перший канал - лівий, другий - правий
            stereo_array = np.empty((n_frames * 2,), dtype=dtype)
            stereo_array[0::2] = left_array
            stereo_array[1::2] = right_array

            # Запис у новий WAV файл
            with wave.open(output_filepath, 'wb') as stereo_wav:
                stereo_wav.setnchannels(2)
                stereo_wav.setsampwidth(sampwidth)
                stereo_wav.setframerate(framerate)
                stereo_wav.writeframes(stereo_array.tobytes())

        print(f"Stereo WAV файл збережено: {output_filepath}")


# Example usage functions
def list_audio_devices():
    """List all available audio input devices"""
    print("Available audio input devices:")
    print("-" * 50)

    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Device {i}: {device['name']}")
            print(f"  Max input channels: {device['max_input_channels']}")
            print(f"  Default sample rate: {device['default_samplerate']}")
            print()
            input_devices.append(i)

    return input_devices


def create_dual_mic_setup(save_files=True, output_directory="recordings"):
    """Example of how to set up dual microphone recording with file storage"""

    # List available devices
    input_devices = list_audio_devices()

    if len(input_devices) < 2:
        print("Error: Need at least 2 input devices")
        return None

    # Select two devices
    device1 = input_devices[1]
    device2 = input_devices[2]

    print(f"Setting up dual microphone with devices {device1} and {device2}")

    # Get device info to determine compatible sample rate
    devices = sd.query_devices()
    device1_info = devices[device1]
    device2_info = devices[device2]

    # Use the device's default sample rate, or find a common one
    samplerate1 = int(device1_info['default_samplerate'])
    samplerate2 = int(device2_info['default_samplerate'])

    # Use the higher sample rate that both devices can support
    # Most devices support both 44100 and 48000
    if samplerate1 == samplerate2:
        chosen_samplerate = samplerate1
    else:
        # Try common sample rates
        common_rates = [48000, 44100, 96000, 22050, 16000]
        chosen_samplerate = 44100  # fallback

        for rate in common_rates:
            try:
                # Test if both devices support this rate
                sd.check_input_settings(device=device1, channels=1, samplerate=rate)
                sd.check_input_settings(device=device2, channels=1, samplerate=rate)
                chosen_samplerate = rate
                break
            except:
                continue

    print(f"Using sample rate: {chosen_samplerate} Hz")
    print(f"Device {device1} default rate: {samplerate1} Hz")
    print(f"Device {device2} default rate: {samplerate2} Hz")

    if save_files:
        print(f"Audio files will be saved to: {output_directory}/")

    # Initialize the synchronizer with file storage
    synchronizer = MultiMicSynchronizer(
        device_list=[device1, device2],
        samplerate=chosen_samplerate,
        blocksize=1024,
        save_to_file=save_files,
        output_dir=output_directory
    )

    return synchronizer

if __name__ == "__main__":
    # Example usage with file storage
    synchronizer = create_dual_mic_setup(save_files=True, output_directory="my_recordings")
    if synchronizer:
        print("\nRecording info:", synchronizer.get_recording_info())
        synchronizer.start_recording()