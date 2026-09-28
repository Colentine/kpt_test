"""Convert the supplied HTML/MathML questions to text without image inputs."""

from dataclasses import dataclass, field
from html.parser import HTMLParser
import re


@dataclass
class Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


class QuestionHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in {"img", "br", "hr", "meta", "link", "input", "source", "wbr"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, dict(attrs)))

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def math_text(node):
    if isinstance(node, str):
        return node.strip()
    tag = node.tag
    children = [c for c in node.children if not isinstance(c, str) or c.strip()]
    parts = [math_text(c) for c in children]
    joined = "".join(parts)
    if tag in {"annotation", "annotation-xml", "none", "mprescripts", "malignmark"}:
        return ""
    if tag == "semantics":
        return parts[0] if parts else ""
    if tag == "mfrac" and len(parts) == 2:
        return r"\frac{" + parts[0] + "}{" + parts[1] + "}"
    if tag in {"msup", "msub"} and len(parts) == 2:
        return "{" + parts[0] + "}" + ("^" if tag == "msup" else "_") + "{" + parts[1] + "}"
    if tag == "msubsup" and len(parts) == 3:
        return "{" + parts[0] + "}_{" + parts[1] + "}^{" + parts[2] + "}"
    if tag == "msqrt":
        return r"\sqrt{" + joined + "}"
    if tag == "mroot" and len(parts) == 2:
        return r"\sqrt[" + parts[1] + "]{" + parts[0] + "}"
    if tag in {"mover", "munder"} and len(parts) == 2:
        command = r"\overset" if tag == "mover" else r"\underset"
        return command + "{" + parts[1] + "}{" + parts[0] + "}"
    if tag == "munderover" and len(parts) == 3:
        return "{" + parts[0] + "}_{" + parts[1] + "}^{" + parts[2] + "}"
    if tag == "mfenced":
        separators = node.attrs.get("separators", ",").replace(" ", "")
        body = parts[0] if parts else ""
        for i, part in enumerate(parts[1:]):
            body += (separators[min(i, len(separators) - 1)] if separators else "") + part
        return node.attrs.get("open", "(") + body + node.attrs.get("close", ")")
    if tag == "mtable":
        return r"\begin{matrix}" + r" \\ ".join(parts) + r"\end{matrix}"
    if tag in {"mtr", "mlabeledtr"}:
        return " & ".join(parts)
    if tag == "mtext":
        return r"\text{" + joined + "}"
    if tag == "mspace":
        return " "
    if tag == "menclose":
        notation = node.attrs.get("notation", "longdiv")
        if notation == "updiagonalstrike":
            return r"\not\subset" if joined == "⊂" else r"\cancel{" + joined + "}"
        if notation == "box":
            return r"\boxed{" + joined + "}"
        raise ValueError(f"Unsupported MathML enclosure: {notation}")
    if tag == "mmultiscripts" and parts:
        before, after = "", ""
        target = "after"
        index = 1
        while index < len(children):
            if isinstance(children[index], Node) and children[index].tag == "mprescripts":
                target = "before"
                index += 1
                continue
            if index + 1 >= len(parts):
                raise ValueError("Unpaired MathML multiscript")
            pair = "_{" + parts[index] + "}^{" + parts[index + 1] + "}"
            if target == "before":
                before += "{}" + pair
            else:
                after += pair
            index += 2
        return before + "{" + parts[0] + "}" + after
    known = {"math", "mrow", "mstyle", "mi", "mn", "mo", "mtd", "mpadded", "mphantom"}
    if tag not in known:
        raise ValueError(f"Unsupported MathML element: {tag}")
    if tag == "mi" and joined in {"ln", "log", "sin", "cos", "tan", "cot", "sec", "csc", "lim", "max", "min"}:
        return "\\" + joined + " "
    return joined


def render_text(node):
    if isinstance(node, str):
        return node
    if node.tag in {"img", "script", "style"} or "exam-foot" in node.attrs.get("class", "").split():
        return ""
    if node.tag == "math":
        return "$" + math_text(node) + "$"
    result = "".join(render_text(child) for child in node.children)
    if node.tag in {"div", "p", "br", "li", "tr"}:
        return result + "\n"
    if node.tag in {"td", "th"}:
        return result + " "
    return result


def normalize_question_text(text):
    # Avoid interpreting plain inequalities as HTML tags.
    if re.search(r"</?(?:div|p|span|img|math|b|table|br)\b", text, flags=re.I):
        parser = QuestionHTML()
        parser.feed(text)
        text = render_text(parser.root)
    text = re.sub(r"\b(?:question|analysis)_\d+-image_\d+\b", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()
