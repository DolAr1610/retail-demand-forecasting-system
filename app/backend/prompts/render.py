from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


PROMPTS_DIR = Path(__file__).resolve().parent

jinja_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(default_for_string=False, disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, **kwargs) -> str:
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs).strip()