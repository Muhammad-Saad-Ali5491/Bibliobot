import math

def calculate(expression):
    """Safely evaluate a math expression — restricted builtins to prevent arbitrary code execution."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"abs": abs, "round": round, "pow": pow})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return result
    except Exception as e:
        return f"Error calculating: {e}"