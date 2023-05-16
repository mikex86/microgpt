from datetime import time

import pyglet

from docker_terminal_provider import DockerTerminalProvider
from terminal_gui import TerminalGui

TERMINAL_WIDTH = 40
TERMINAL_HEIGHT = 20

if __name__ == '__main__':
    term_provider = DockerTerminalProvider('test_ubuntu', (TERMINAL_WIDTH, TERMINAL_HEIGHT))
    # term_provider.send_input('apt update && apt install vim -y\r')
    pyglet.clock.schedule_once(lambda dt: term_provider.send_input('vim test.txt\r'), 1)
    pyglet.clock.schedule_once(lambda dt: term_provider.send_input(
        """
#include <stdio.h>
int main() {
    char operation;
    double num1, num2, result;

    printf("Enter an operator (+, -, *): ");
    scanf(" %c", &operation);

    printf("Enter two numbers: ");
    scanf("%lf %lf", &num1, &num2);

    switch (operation) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        default:
            printf("Invalid operator! Please use +, -, or *.");
            return 1;
    }

    printf("%.2lf %c %.2lf = %.2lf\\n", num1, operation, num2, result);
    return 0;
}
        """.replace('\n', '\r')), 3)

    # Save and exit
    pyglet.clock.schedule_once(lambda dt: term_provider.send_input('\x1b:wq\r'), 4)
    term_gui = TerminalGui('Terminal', term_provider, True)

    pyglet.clock.schedule(lambda dt: term_provider.update())

    pyglet.app.run()
