import os
import elevenlabs
from elevenlabs.client import ElevenLabs
import subprocess
from pathlib import Path
import platform
from dotenv import load_dotenv


load_dotenv()


ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio=client.text_to_speech.convert(
        text= input_text,
        voice_id="nPczCjzI2devNBz1zQrb", #"JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format= "mp3_22050_32",
    )
    elevenlabs.save(audio, output_filepath)
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            #subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
            """subprocess.run([
                    'powershell', '-c',f'$player = New-Object -ComObject WMPlayer.OCX; '
                    f'$media = $player.newMedia("{output_filepath}"); '
                    f'$player.currentPlaylist.appendItem($media); '
                    f'$player.controls.play(); Start-Sleep -Seconds 10'
                ])"""
            """path_uri = Path(output_filepath).resolve().as_uri()  # Converts to file:///C:/... format
            powershell_command = (
                f"$player = New-Object -ComObject WMPlayer.OCX; "
                f"$media = $player.newMedia('{path_uri}'); "
                f"$player.controls.play(); "
                f"Start-Sleep -Seconds 5;"
            )
            print("▶️ Playing:", path_uri)  # Debug
            os.startfile(output_filepath)
            subprocess.run(["powershell", "-Command", powershell_command])"""
            subprocess.run([
            "powershell", "-c",
            f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'
        ])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


from gtts import gTTS

def text_to_speech_with_gtts(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            #subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
            """subprocess.run([
                    'powershell', '-c',
                    f'$player = New-Object -ComObject WMPlayer.OCX; '
                    f'$media = $player.newMedia("{output_filepath}"); '
                    f'$player.currentPlaylist.appendItem($media); '
                    f'$player.controls.play(); Start-Sleep -Seconds 10'
                ])"""
            

        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


#input_text = "Hi, I am doing fine, how are you? This is a acchuuuuuu"
#output_filepath = "test_text_to_speech_check.mp3"
#text_to_speech_with_elevenlabs(input_text, output_filepath)
#text_to_speech_with_gtts(input_text, output_filepath)