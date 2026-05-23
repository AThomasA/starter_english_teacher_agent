# ui/app.py — redireciona para app_md.py (código ativo)
# Para rodar: streamlit run ui/app.py  OU  streamlit run ui/app_md.py
from pathlib import Path

_app_md = Path(__file__).parent / "app_md.py"
exec(compile(_app_md.read_text(encoding="utf-8"), str(_app_md), "exec"))
