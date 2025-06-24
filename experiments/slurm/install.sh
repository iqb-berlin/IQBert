  GNU nano 7.2                                                                                        install.sh
#!/bin/bash
module load python/3.12.3
python -m venv ~/myenv
source ~/myenv/bin/activate
pip install transformers datasets torch transformers[torch]
