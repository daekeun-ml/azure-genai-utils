import os
import json
import glob
import html
import datetime
import azure.cognitiveservices.speech as speechsdk
from azure_genai_utils.aoai import AOAI
from typing import Optional

STT_LOCALE_DICT = {
    "Afrikaans (South Africa)": "af-ZA",
    "Amharic (Ethiopia)": "am-ET",
    "Arabic (United Arab Emirates)": "ar-AE",
    "Arabic (Bahrain)": "ar-BH",
    "Arabic (Algeria)": "ar-DZ",
    "Arabic (Egypt)": "ar-EG",
    "Arabic (Israel)": "ar-IL",
    "Arabic (Iraq)": "ar-IQ",
    "Arabic (Jordan)": "ar-JO",
    "Arabic (Kuwait)": "ar-KW",
    "Arabic (Lebanon)": "ar-LB",
    "Arabic (Libya)": "ar-LY",
    "Arabic (Morocco)": "ar-MA",
    "Arabic (Oman)": "ar-OM",
    "Arabic (Palestinian Authority)": "ar-PS",
    "Arabic (Qatar)": "ar-QA",
    "Arabic (Saudi Arabia)": "ar-SA",
    "Arabic (Syria)": "ar-SY",
    "Arabic (Tunisia)": "ar-TN",
    "Arabic (Yemen)": "ar-YE",
    "Assamese (India)": "as-IN",
    "Azerbaijani (Latin, Azerbaijan)": "az-AZ",
    "Bulgarian (Bulgaria)": "bg-BG",
    "Bengali (India)": "bn-IN",
    "Bosnian (Bosnia and Herzegovina)": "bs-BA",
    "Chinese": "zh-CN",
    "Chinese (Wu, Simplified)": "wuu-CN",
    "Chinese (Cantonese, Simplified)": "yue-CN",
    "Chinese (Mandarin, Simplified)": "zh-CN",
    "Chinese (Jilu Mandarin, Simplified)": "zh-CN-shandong",
    "Chinese (Southwestern Mandarin, Simplified)": "zh-CN-sichuan",
    "Chinese (Cantonese, Traditional)": "zh-HK",
    "Chinese (Taiwanese Mandarin, Traditional)": "zh-TW",
    "Catalan": "ca-ES",
    "Czech (Czechia)": "cs-CZ",
    "Welsh (United Kingdom)": "cy-GB",
    "Danish (Denmark)": "da-DK",
    "German": "de-DE",
    "German (Austria)": "de-AT",
    "German (Switzerland)": "de-CH",
    "German (Germany)": "de-DE",
    "Greek (Greece)": "el-GR",
    "English": "en-US",
    "English (Australia)": "en-AU",
    "English (Canada)": "en-CA",
    "English (United Kingdom)": "en-GB",
    "English (Ghana)": "en-GH",
    "English (Hong Kong SAR)": "en-HK",
    "English (Ireland)": "en-IE",
    "English (India)": "en-IN",
    "English (Kenya)": "en-KE",
    "English (Nigeria)": "en-NG",
    "English (New Zealand)": "en-NZ",
    "English (Philippines)": "en-PH",
    "English (Singapore)": "en-SG",
    "English (Tanzania)": "en-TZ",
    "English (United States)": "en-US",
    "English (South Africa)": "en-ZA",
    "Spanish (Argentina)": "es-AR",
    "Spanish (Bolivia)": "es-BO",
    "Spanish (Chile)": "es-CL",
    "Spanish (Colombia)": "es-CO",
    "Spanish (Costa Rica)": "es-CR",
    "Spanish (Cuba)": "es-CU",
    "Spanish (Dominican Republic)": "es-DO",
    "Spanish (Ecuador)": "es-EC",
    "Spanish (Spain)": "es-ES",
    "Spanish (Equatorial Guinea)": "es-GQ",
    "Spanish (Guatemala)": "es-GT",
    "Spanish (Honduras)": "es-HN",
    "Spanish (Mexico)": "es-MX",
    "Spanish (Nicaragua)": "es-NI",
    "Spanish (Panama)": "es-PA",
    "Spanish (Peru)": "es-PE",
    "Spanish (Puerto Rico)": "es-PR",
    "Spanish (Paraguay)": "es-PY",
    "Spanish (El Salvador)": "es-SV",
    "Spanish (United States)": "es-US",
    "Spanish (Uruguay)": "es-UY",
    "Spanish (Venezuela)": "es-VE",
    "Estonian (Estonia)": "et-EE",
    "Basque": "eu-ES",
    "Persian (Iran)": "fa-IR",
    "Finnish (Finland)": "fi-FI",
    "Filipino (Philippines)": "fil-PH",
    "French (Belgium)": "fr-BE",
    "French (Canada)": "fr-CA",
    "French (Switzerland)": "fr-CH",
    "French (France)": "fr-FR",
    "Irish (Ireland)": "ga-IE",
    "Galician": "gl-ES",
    "Gujarati (India)": "gu-IN",
    "Hebrew (Israel)": "he-IL",
    "Hindi (India)": "hi-IN",
    "Croatian (Croatia)": "hr-HR",
    "Hungarian (Hungary)": "hu-HU",
    "Armenian (Armenia)": "hy-AM",
    "Indonesian (Indonesia)": "id-ID",
    "Icelandic (Iceland)": "is-IS",
    "Italian (Switzerland)": "it-CH",
    "Italian (Italy)": "it-IT",
    "Japanese": "ja-JP",
    "Javanese (Latin, Indonesia)": "jv-ID",
    "Georgian (Georgia)": "ka-GE",
    "Kazakh (Kazakhstan)": "kk-KZ",
    "Khmer (Cambodia)": "km-KH",
    "Kannada (India)": "kn-IN",
    "Korean": "ko-KR",
    "Lao (Laos)": "lo-LA",
    "Lithuanian (Lithuania)": "lt-LT",
    "Latvian (Latvia)": "lv-LV",
    "Macedonian (North Macedonia)": "mk-MK",
    "Malayalam (India)": "ml-IN",
    "Mongolian (Mongolia)": "mn-MN",
    "Marathi (India)": "mr-IN",
    "Malay (Malaysia)": "ms-MY",
    "Maltese (Malta)": "mt-MT",
    "Burmese (Myanmar)": "my-MM",
    "Norwegian Bokmål (Norway)": "nb-NO",
    "Nepali (Nepal)": "ne-NP",
    "Dutch (Belgium)": "nl-BE",
    "Dutch (Netherlands)": "nl-NL",
    "Odia (India)": "or-IN",
    "Punjabi (India)": "pa-IN",
    "Polish (Poland)": "pl-PL",
    "Pashto (Afghanistan)": "ps-AF",
    "Portuguese (Brazil)": "pt-BR",
    "Portuguese (Portugal)": "pt-PT",
    "Romanian (Romania)": "ro-RO",
    "Russian (Russia)": "ru-RU",
    "Sinhala (Sri Lanka)": "si-LK",
    "Slovak (Slovakia)": "sk-SK",
    "Slovenian (Slovenia)": "sl-SI",
    "Albanian (Albania)": "sq-AL",
    "Serbian (Cyrillic, Bosnia and Herzegovina)": "sr-BA",
    "Serbian (Cyrillic, Serbia)": "sr-RS",
    "Sundanese (Latin, Indonesia)": "su-ID",
    "Swedish (Sweden)": "sv-SE",
    "Swahili (Kenya)": "sw-KE",
    "Swahili (Tanzania)": "sw-TZ",
    "Tamil (India)": "ta-IN",
    "Tamil (Sri Lanka)": "ta-LK",
    "Tamil (Malaysia)": "ta-MY",
    "Tamil (Singapore)": "ta-SG",
    "Telugu (India)": "te-IN",
    "Thai (Thailand)": "th-TH",
    "Tigrinya (Ethiopia)": "ti-ET",
    "Turkish (Turkey)": "tr-TR",
    "Ukrainian (Ukraine)": "uk-UA",
    "Urdu (India)": "ur-IN",
    "Urdu (Pakistan)": "ur-PK",
    "Uzbek (Latin, Uzbekistan)": "uz-UZ",
    "Vietnam": "vi-VN",
    "Xhosa (South Africa)": "xh-ZA",
    "Yiddish (World)": "yi-001",
    "Yoruba (Nigeria)": "yo-NG",
    "Zulu (South Africa)": "zu-ZA",
}


def get_audio_file_by_speech_synthesis(
    speech_synthesizer, text, file_path, lang, default_tts_voice
):
    ssml = f"""<speak version='1.0'  xmlns="https://www.w3.org/2001/10/synthesis" xml:lang='{lang}'>
                     <voice name='{default_tts_voice}'>
                             {html.escape(text)}
                     </voice>
                   </speak>"""
    speech_sythesis_result = speech_synthesizer.speak_ssml_async(ssml).get()
    stream = speechsdk.AudioDataStream(speech_sythesis_result)
    stream.save_to_wav_file(file_path)


class CustomSpeechToTextGenerator(AOAI):
    """
    Custom Speech to Text Generator class
    """

    def __init__(
        self,
        ai_speech_api_key: Optional[str] = None,
        ai_speech_region: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Custom Speech to Text Generator
        """
        super().__init__()

        if ai_speech_api_key is None:
            ai_speech_api_key = os.getenv("AZURE_AI_SPEECH_API_KEY")

        if ai_speech_region is None:
            ai_speech_region = os.getenv("AZURE_AI_SPEECH_REGION")

        self.ai_speech_api_key = ai_speech_api_key
        self.ai_speech_region = ai_speech_region

        try:
            # Initialize SpeechConfig
            speech_config = speechsdk.SpeechConfig(
                subscription=self.ai_speech_api_key, region=self.ai_speech_region
            )
            if not speech_config:
                raise ValueError(
                    "SpeechConfig initialization failed. Check your API key and region."
                )

            # Initialize AudioConfig
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            if not audio_config:
                raise ValueError(
                    "AudioConfig initialization failed. Check your audio configuration."
                )

            # Initialize SpeechSynthesizer
            self.speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config
            )
            if not self.speech_synthesizer:
                raise ValueError("SpeechSynthesizer initialization failed.")

            print("=== Initialized CustomSpeechToTextGenerator ===")

        except Exception as e:
            print(f"An error occurred during Speech SDK initialization: {e}")
            # You can choose to log the error or raise it depending on your application's needs
            raise

    @staticmethod
    def print_supported_languages():
        """
        Print supported languages
        """
        print(list(STT_LOCALE_DICT.keys()))

    def generate_synthetic_text(
        self,
        synthetic_text_file: str = "cc_support_expressions.jsonl",
        num_samples: int = 2,
        custom_speech_lang: str = "Korean",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """
        Generate QnA for custom speech languages
        """
        if custom_speech_lang in STT_LOCALE_DICT:
            self.custom_speech_locale = STT_LOCALE_DICT[custom_speech_lang]
        else:
            raise ValueError(
                f"Unsupported language '{custom_speech_lang}'. Please choose from: {list(STT_LOCALE_DICT.keys())}"
            )
        self.synthetic_text_file = synthetic_text_file

        topic = f"""
        Call center QnA related expected spoken utterances for {custom_speech_lang} and English languages.
        """
        question = f"""
        create {num_samples} lines of jsonl of the topic in {custom_speech_lang} and English. jsonl format is required. 
        use 'no' as number and '{self.custom_speech_locale}', 'en-US' keys for the languages.
        only include the lines as the result. Do not include ```jsonl, ``` and blank line in the result. 
        """

        system_message = """
        Generate plain text sentences of #topic# related text to improve the recognition of domain-specific words and phrases.
        Domain-specific words can be uncommon or made-up words, but their pronunciation must be straightforward to be recognized. 
        Use text data that's close to the expected spoken utterances. The nummber of utterances per line should be 1. 
        Here is examples of the expected format:
        {"no": 1, "string": "string", "string": "string"}
        {"no": 2, "string": "string", "string": "string"}
        """

        user_message = f"""
        #topic#: {topic}
        Question: {question}
        """
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        print("Usage Information:")
        if response.usage:
            # print(f"Cached Tokens: {response.usage.prompt_tokens_details.cached_tokens}") #only o1 models support this
            print(f"Completion Tokens: {response.usage.completion_tokens}")
            print(f"Prompt Tokens: {response.usage.prompt_tokens}")
            print(f"Total Tokens: {response.usage.total_tokens}")
            content = response.choices[0].message.content
            # print(content)
            with open(self.synthetic_text_file, "w", encoding="utf-8") as f:
                for line in content.split("\n"):
                    if line.strip():  # Check if the line is not empty
                        f.write(line + "\n")
            return content
        else:
            print("No usage information available.")
            return None

    def save_synthetic_text(
        self,
        output_dir: str = "plain_text",
        delete_old_data: bool = True,
    ) -> None:
        """
        Save synthetic text to a file
        """
        # Check https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt for supported locale
        languages = [
            self.custom_speech_locale
        ]  # List of languages to generate audio files

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if delete_old_data:
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))

        with open(self.synthetic_text_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    expression = json.loads(line)
                    for lang in languages:
                        text = expression[lang]
                        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        file_name = f"{lang}_{timestamp}.txt"
                        with open(
                            os.path.join(output_dir, file_name), "a", encoding="utf-8"
                        ) as plain_text:
                            plain_text.write(f"{text}\n")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(e)

        plain_texts = glob.glob(f"{output_dir}/*.txt")

        for file_name in plain_texts:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                print(f"=== File Name: {file_name} ===")
                print(content)

    def generate_synthetic_wav(
        self, output_dir="synthetic_data", delete_old_data: bool = True
    ):
        """
        Generate synthetic audio files
        """
        # Check https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt for supported locale
        language = (
            self.custom_speech_locale
        )  # List of languages to generate audio files

        TTS_FOR_TRAIN = os.getenv("TTS_FOR_TRAIN")
        TTS_FOR_EVAL = os.getenv("TTS_FOR_EVAL")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if delete_old_data:
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))

        train_tts_voices = TTS_FOR_TRAIN.split(",")

        for tts_voice in train_tts_voices:
            with open(self.synthetic_text_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        expression = json.loads(line)
                        no = expression["no"]
                        text = expression[language]
                        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        file_name = f"{no}_locale_{language}_speaker_{tts_voice}_{timestamp}.wav"
                        print(f"Generating {file_name}")
                        get_audio_file_by_speech_synthesis(
                            self.speech_synthesizer,
                            text,
                            os.path.join(output_dir, file_name),
                            language,
                            tts_voice,
                        )
                        with open(
                            f"{output_dir}/manifest.txt", "a", encoding="utf-8"
                        ) as manifest_file:
                            manifest_file.write(f"{file_name}\t{text}\n")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {line}")
                        print(e)
