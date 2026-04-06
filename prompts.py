GENERATION_TEMPLATE = """You are a professional 3D sketch artist.

Generate a 3D object as a list of cubic Bezier curves.

Output rules:
1. Return exactly one Python list wrapped by <curves> and </curves>.
2. The top-level object must be a list of curves.
3. Each curve must contain exactly 4 control points.
4. Each control point must be a list of exactly 3 numbers: [x, y, z].
5. Do not include comments, prose, variable assignments, or markdown inside the tags.

Geometry rules:
1. Use a right-handed coordinate system.
2. +Z is up, +X is right, +Y is forward.
3. Keep the object centered near the origin.
4. Keep coordinates roughly inside [-0.8, 0.8].
5. Use depth variation. Avoid collapsing everything onto one plane.
6. Use enough curves to capture the main structure, but avoid redundant tiny curves.

Valid format example:
<curves>
[
  [[-0.5, -0.5, -0.4], [-0.15, -0.5, -0.4], [0.15, -0.5, -0.4], [0.5, -0.5, -0.4]],
  [[-0.5, 0.5, -0.4], [-0.15, 0.5, -0.4], [0.15, 0.5, -0.4], [0.5, 0.5, -0.4]],
  [[-0.5, -0.5, 0.4], [-0.15, -0.5, 0.4], [0.15, -0.5, 0.4], [0.5, -0.5, 0.4]],
  [[-0.5, 0.5, 0.4], [-0.15, 0.5, 0.4], [0.15, 0.5, 0.4], [0.5, 0.5, 0.4]],
  [[-0.5, -0.5, -0.4], [-0.5, -0.2, -0.1], [-0.5, 0.2, 0.1], [-0.5, 0.5, 0.4]],
  [[0.5, -0.5, -0.4], [0.5, -0.2, -0.1], [0.5, 0.2, 0.1], [0.5, 0.5, 0.4]]
]
</curves>
"""


EXPERIENCE_TEMPLATE = """You are reviewing two 3D sketch generations for the same text prompt.

Prompt:
{theme}

Existing experiences:
{existing_experiences}

Better result score: {best_score:.4f}
Better result:
{best_response}

Worse result score: {worst_score:.4f}
Worse result:
{worst_response}

Write up to {max_new_experiences} short reusable experiences that explain why the better result is better.

Rules:
1. Focus on 3D drawing strategy, geometry, structure, depth, symmetry, or control-point usage.
2. Write reusable lessons, not prompt-specific descriptions.
3. Do not repeat existing experiences.
4. Each item should be one concise sentence.
5. Return only a JSON list of strings.
"""


def format_experiences(experiences):
    if not experiences:
        return "None"
    return "\n".join(f"- {item}" for item in experiences)


def build_generation_prompt(theme, experiences):
    sections = [GENERATION_TEMPLATE]
    if experiences:
        sections.append("Useful experiences:\n" + format_experiences(experiences))
    sections.append(f"Theme: {theme}\n\nReturn the final result now.")
    return "\n\n".join(sections)


def build_experience_prompt(
    theme,
    best_response,
    best_score,
    worst_response,
    worst_score,
    experiences,
    max_new_experiences,
):
    return EXPERIENCE_TEMPLATE.format(
        theme=theme,
        existing_experiences=format_experiences(experiences),
        best_response=best_response,
        best_score=best_score,
        worst_response=worst_response,
        worst_score=worst_score,
        max_new_experiences=max_new_experiences,
    )
