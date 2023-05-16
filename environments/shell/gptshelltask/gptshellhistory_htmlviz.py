from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from environments.shell.gptshelltask.gptshelltask_executor import ShellHistoryEntry


def save_shell_task_history_as_html(entries: List['ShellHistoryEntry'], html_path: str):
    with open(html_path, 'w', encoding="utf-8") as f:
        # Write HTML header
        f.write(
            """<html>
<head>
<style>html {
    font-family: Arial, sans-serif;
    font-size: 16px;
}

pre {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 3px;
    font-family: Consolas, monospace;
    font-size: 14px;
    margin: 0;
    padding: 10px;
    white-space: pre-wrap;
}

h4 {
    font-size: 15px;
    font-weight: bold;
    margin: 5px 0 5px 0;
}

p {
    margin-top: 0;
}
</style>
</head>
<body>
            """
        )

        # Write history entries
        for entry in entries:
            f.write("<div style='border: 1.5px solid #333; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>")

            f.write("<h4>Terminal Context:</h4>")
            f.write("<pre>{}</pre>".format(entry.terminal_context.replace("<", "&lt").replace(">", "&gt")))

            if not entry.erroneous_completion:
                if entry.observation:
                    f.write("<h4>Observation</h4>")
                    if entry.ctx_is_few_shot_learning_helper:
                        f.write("<p>{}</p>".format(entry.observation))
                    else:
                        f.write("<p>Omitted</p>")

                if entry.thought:
                    f.write("<h4>Thought</h4>")
                    f.write("<p>{}</p>".format(entry.thought))

                if entry.response_stdin:
                    f.write("<h4>Response Stdin:</h4>")
                    f.write("<pre>{}</pre>".format(entry.response_stdin.replace("<", "&lt").replace(">", "&gt")))

                f.write("<h4>Delay (seconds):</h4>")
                f.write('<input type="number" id="quantity" name="quantity" value={} disabled>'.format(entry.delay))
            else:
                f.write("<h4>Erroneous completion</h4>")
                f.write("<pre>{}</pre>".format(entry.erroneous_completion.replace("<", "&lt").replace(">", "&gt")))

                if entry.error_message:
                    f.write("<h4>Error Message</h4>")
                    f.write("<p>{}</p>".format(entry.error_message.replace("<", "&lt").replace(">", "&gt")))

            f.write("</div>")

        # Write HTML footer
        f.write("</body></html>")
