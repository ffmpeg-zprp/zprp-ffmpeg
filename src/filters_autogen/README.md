# Automatyczne wykrywanie filtrów

Ta część kodu nie należy do paczki, generuje część kodu na podstawie kodu źródłowego FFmpeg.

Aby wygenerować filtry należy sklonować repozytorium pycparser (potrzebne są dodatkowe nagłówki) oraz ffmpeg:
```
git clone https://github.com/FFmpeg/FFmpeg.git
git clone https://github.com/eliben/pycparser.git
```

Następnie usunąć/zmienić nazwę `libavutil/thread.h` (TODO: czy jest lepsze rozwiązanie?)

```
rm FFMpeg/libavutil/thread.h
```



Następnie uruchomić `generate_filters.py`
