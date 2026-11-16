"""kortexa's exact 8K needle prompt ids (upstream/runtime.py encode_prompt): the checkpoint's own chat template
(auxiliary/chat_template.jinja) rendered HF-style for one user message with reasoning_effort "none" and
add_generation_prompt False, then "<|message_model|><|content_text|>" appended, encoded with tokenizer.json and no
special tokens added. Expect 8,128 ids (validation/portable-spark-8k.json: prompt_tokens 8128).
  uv run --no-project --with jinja2==3.1.2 --with tokenizers==0.21.1 python -I tests/needle/make_kortexa_ids.py CKPT OUT
"""
import datetime, json, sys
import jinja2
from tokenizers import Tokenizer

ckpt, out = sys.argv[1], sys.argv[2]
prompt = open(f"{ckpt}/validation/portable-spark-8k-prompt.txt").read()


def raise_exception(m):
    raise ValueError(m)


env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
text = env.from_string(open(f"{ckpt}/auxiliary/chat_template.jinja").read()).render(
    messages=[{"role": "user", "content": prompt}], reasoning_effort="none", add_generation_prompt=False)
text += "<|message_model|><|content_text|>"
ids = Tokenizer.from_file(f"{ckpt}/auxiliary/tokenizer.json").encode(text, add_special_tokens=False).ids
json.dump(ids, open(out, "w"))
print(f"{len(ids)} ids -> {out}")
sys.exit(0 if len(ids) == 8128 else 1)
