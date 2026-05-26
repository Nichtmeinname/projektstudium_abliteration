Abliteration - Entfernen der Refusal Direction

In diesem Projekt geht es um das Entfernen der Ablehnungsrichtung (Refusal Direction).
Dabei geht es darum, das Verhalten einens LLMs so zu verändern, dass schädliche, illegale
oder bedrohliche Prompts nicht mehr abgelehnt werden.
Durch das Entfernen dieses Ablehnungsstroms sollen diese Prompts beantwortet und nicht verweigert werden.

Die verwendeten Modelle stammen von Huggingface. Um die Programme durchzuführen, wird eine
Huggingface Account benötigt. Außerdem muss eine Accesstoken bei Huggingface gesetzt und
anschließend als Umgebungsvariable gesetzt werden (https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken).

Verwendete Python-Version: 3.11

Verwendete Modelle:
- Qwen2.5 mit 3B Parametern
- Qwen2.5 mit 7B Parametern

Hardwarevoraussetzungen:
- GPU
  - Bei 3B Parametern wird mindestens eine Grafikkarte mit 6 GB benötigt.
  - Bei 7B Parametern wird mindestens eine Grafikkarte mit 16 GB benötigt.
- CPU
  - Bei 3B Parametern wird mindestens 8 GB RAM benötigt.
  - Bei 7B Parametern wird mindestens 32 GB RAM benötigt.