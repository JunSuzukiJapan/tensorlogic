#!/usr/bin/env python3
"""
TensorLogic Jupyter Kernel

Implements the Jupyter kernel protocol for TensorLogic,
enabling interactive notebook execution with syntax highlighting,
tab completion, and rich output support.
"""

from ipykernel.kernelbase import Kernel
from ipykernel.ipkernel import IPythonKernel
import re
import sys
from typing import Dict, List, Optional, Any

try:
    from tensorlogic import Interpreter
except ImportError:
    # Fallback for development
    sys.path.insert(0, '../..')
    from tensorlogic import Interpreter

__version__ = '0.1.0'


class TensorLogicKernel(Kernel):
    """Jupyter kernel for TensorLogic language"""

    implementation = 'TensorLogic'
    implementation_version = __version__
    language = 'tensorlogic'
    language_version = '0.1.0'
    language_info = {
        'name': 'tensorlogic',
        'mimetype': 'text/x-tensorlogic',
        'file_extension': '.tl',
        'codemirror_mode': 'tensorlogic',
        'pygments_lexer': 'tensorlogic',
    }
    banner = "TensorLogic - A unified tensor algebra and logic programming language"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = Interpreter()
        self._execution_count = 0

    def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, Any]] = None,
        allow_stdin: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute TensorLogic code in the kernel.

        Args:
            code: TensorLogic source code to execute
            silent: If True, don't send output to frontend
            store_history: Whether to store this execution in history
            user_expressions: Additional expressions to evaluate
            allow_stdin: Whether to allow input from user

        Returns:
            Execution result dictionary
        """
        if not code.strip():
            return {
                'status': 'ok',
                'execution_count': self._execution_count,
                'payload': [],
                'user_expressions': {},
            }

        self._execution_count += 1

        try:
            # Capture output
            import io
            from contextlib import redirect_stdout, redirect_stderr

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = self.interpreter.execute(code)

            # Get captured output
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            # Send output to frontend
            if not silent:
                if stdout_text:
                    self.send_response(
                        self.iopub_socket,
                        'stream',
                        {'name': 'stdout', 'text': stdout_text}
                    )
                if stderr_text:
                    self.send_response(
                        self.iopub_socket,
                        'stream',
                        {'name': 'stderr', 'text': stderr_text}
                    )

                # Send execution result if available
                if result:
                    self.send_response(
                        self.iopub_socket,
                        'execute_result',
                        {
                            'execution_count': self._execution_count,
                            'data': {'text/plain': str(result)},
                            'metadata': {}
                        }
                    )

            return {
                'status': 'ok',
                'execution_count': self._execution_count,
                'payload': [],
                'user_expressions': {},
            }

        except SyntaxError as e:
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    'error',
                    {
                        'ename': 'SyntaxError',
                        'evalue': str(e),
                        'traceback': [str(e)]
                    }
                )

            return {
                'status': 'error',
                'execution_count': self._execution_count,
                'ename': 'SyntaxError',
                'evalue': str(e),
                'traceback': [str(e)]
            }

        except Exception as e:
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    'error',
                    {
                        'ename': type(e).__name__,
                        'evalue': str(e),
                        'traceback': [str(e)]
                    }
                )

            return {
                'status': 'error',
                'execution_count': self._execution_count,
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [str(e)]
            }

    def do_complete(self, code: str, cursor_pos: int) -> Dict[str, Any]:
        """
        Provide tab completion suggestions.

        Args:
            code: Current code in the cell
            cursor_pos: Cursor position in the code

        Returns:
            Completion result dictionary
        """
        # Extract word at cursor
        code_before_cursor = code[:cursor_pos]
        word_match = re.search(r'(\w+)$', code_before_cursor)

        if not word_match:
            return {
                'status': 'ok',
                'matches': [],
                'cursor_start': cursor_pos,
                'cursor_end': cursor_pos,
                'metadata': {}
            }

        word = word_match.group(1)
        cursor_start = cursor_pos - len(word)

        # TensorLogic keywords and built-in functions
        keywords = [
            'tensor', 'relation', 'rule', 'main', 'query', 'embedding',
            'python', 'import', 'as', 'call',
            'float16', 'int16', 'int32', 'int64', 'bool', 'complex16',
            'if', 'else', 'for', 'while', 'return',
            'print', 'assert',
        ]

        # Get variable names from interpreter
        try:
            variables = self.interpreter.list_variables()
        except:
            variables = []

        # Combine all suggestions
        all_suggestions = keywords + variables

        # Filter based on current word
        matches = [s for s in all_suggestions if s.startswith(word)]

        return {
            'status': 'ok',
            'matches': sorted(matches),
            'cursor_start': cursor_start,
            'cursor_end': cursor_pos,
            'metadata': {}
        }

    def do_inspect(self, code: str, cursor_pos: int, detail_level: int = 0) -> Dict[str, Any]:
        """
        Provide documentation/inspection for objects.

        Args:
            code: Current code
            cursor_pos: Cursor position
            detail_level: Level of detail (0 or 1)

        Returns:
            Inspection result dictionary
        """
        # Extract word at cursor
        code_before_cursor = code[:cursor_pos]
        word_match = re.search(r'(\w+)$', code_before_cursor)

        if not word_match:
            return {'status': 'ok', 'found': False, 'data': {}, 'metadata': {}}

        word = word_match.group(1)

        # Documentation for TensorLogic keywords
        docs = {
            'tensor': 'Declare a tensor variable with shape and type',
            'python': 'Python integration: import modules or call functions',
            'import': 'Import Python modules: python import module as alias',
            'call': 'Call Python functions: python.call("function", args)',
            'float16': 'Half-precision (16-bit) floating point type',
            'main': 'Main execution block',
            'print': 'Print values to output',
        }

        doc = docs.get(word)
        if doc:
            return {
                'status': 'ok',
                'found': True,
                'data': {'text/plain': doc},
                'metadata': {}
            }

        # Check if it's a variable
        try:
            var_value = self.interpreter.get_variable(word)
            if var_value is not None:
                var_info = f"Variable: {word}\nType: {type(var_value).__name__}"
                if hasattr(var_value, 'shape'):
                    var_info += f"\nShape: {var_value.shape}"

                return {
                    'status': 'ok',
                    'found': True,
                    'data': {'text/plain': var_info},
                    'metadata': {}
                }
        except:
            pass

        return {'status': 'ok', 'found': False, 'data': {}, 'metadata': {}}

    def do_shutdown(self, restart: bool) -> Dict[str, Any]:
        """
        Shutdown the kernel.

        Args:
            restart: Whether kernel will restart after shutdown

        Returns:
            Shutdown result dictionary
        """
        return {'status': 'ok', 'restart': restart}


if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=TensorLogicKernel)
