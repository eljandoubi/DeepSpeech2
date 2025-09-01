# DeepSpeech2
Building the DeepSpeech2

* Clone the repository and Go to PaliGemma directory.
```bash
git clone https://github.com/eljandoubi/DeepSpeech2.git && cd DeepSpeech2
```

* Build environment.
```bash
uv sync
```

* Donwload Data
```bash
chmod +x download_librispeech.sh
./download_librispeech.sh ./data
```
* Train the model

```bash
python src/train.py
```