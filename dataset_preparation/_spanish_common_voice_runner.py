from dataset_preparation.dataset_downloader import download_and_extract_data
from transformers.spanish_common_voice import CommonVoiceKaldiTransformer

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_url='https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/es.tar.gz',
        dataset_name='CommonVoiceSpanish',
        download_folder="/home/stanislav/y-data/gong/Domain-specific-ESPnet/data")
    print("Dataset location:", dataset_location)

    transformer = CommonVoiceKaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory='/home/stanislav/y-data/gong/Domain-specific-ESPnet/espnet_emulation/egs/common_voice/asr1')