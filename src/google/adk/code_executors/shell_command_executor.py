# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shell command executor for ADK."""

import asyncio
import logging
import shlex
import subprocess
from typing import List, Optional, Tuple, Dict, Any, Union

from typing_extensions import override

from .base_code_executor import BaseCodeExecutor, CodeExecutionResult

logger = logging.getLogger(__name__)

class ShellCommandExecutor(BaseCodeExecutor):
    """Executes shell commands directly using asyncio subprocess."""

    def __init__(self, timeout: int = 60):
        """Initialize the ShellCommandExecutor.
        
        Args:
            timeout: Default timeout in seconds for command execution
        """
        self.timeout = timeout

    @override
    async def execute_code(
        self, *, code: str, context: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionResult:
        """Execute a shell command.
        
        Args:
            code: The shell command to execute
            context: Optional execution context
            
        Returns:
            CodeExecutionResult with command output
        """
        try:
            # If code contains multiple lines, join them with &&
            command = code.strip()
            if '\n' in command:
                command = ' && '.join([line.strip() for line in command.split('\n') if line.strip()])
            
            # Execute the command
            stdout, stderr, returncode = await self._execute_command(command)
            
            # Format the result
            output = f"Exit Code: {returncode}\n"
            if stdout:
                output += f"Standard Output:\n{stdout}\n"
            if stderr:
                output += f"Standard Error:\n{stderr}\n"
                
            return CodeExecutionResult(
                success=(returncode == 0),
                output=output,
                error=stderr if returncode != 0 else None
            )
        except Exception as e:
            logger.error(f"Error executing shell command: {e}")
            return CodeExecutionResult(
                success=False,
                output="",
                error=f"Error executing shell command: {e}"
            )
    
    async def execute(
        self, command: Union[str, List[str]], timeout: Optional[int] = None
    ) -> Tuple[str, str, int]:
        """Execute a shell command directly.
        
        Args:
            command: The command to execute as a string or list of strings
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        if isinstance(command, str):
            command_str = command
        else:
            command_str = ' '.join(command)
            
        return await self._execute_command(command_str, timeout)
            
    async def _execute_command(
        self, command: str, timeout: Optional[int] = None
    ) -> Tuple[str, str, int]:
        """Execute a shell command using asyncio subprocess.
        
        Args:
            command: The command as a string
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        timeout = timeout or self.timeout
        
        try:
            # Split the command string using shlex to handle quoted arguments properly
            cmd_parts = shlex.split(command)
            
            # Use asyncio.subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                # Wait for the process with timeout
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                return stdout.decode().strip(), stderr.decode().strip(), process.returncode
            except asyncio.TimeoutError:
                # Try to terminate the process on timeout
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()  # Force kill if terminate doesn't work
                    await process.wait()
                
                err_msg = f"Error: Command timed out after {timeout}s: {command}"
                logger.warning(err_msg)
                return "", err_msg, -2  # Indicate timeout with -2
            
        except FileNotFoundError:
            cmd_name = cmd_parts[0] if cmd_parts else command
            err_msg = f"Error: Command '{cmd_name}' not found. Is it installed and in PATH?"
            logger.error(err_msg)
            return "", err_msg, -1  # Indicate file not found with -1
        except Exception as e:
            err_msg = f"Unexpected error running command {command}: {e}"
            logger.error(err_msg)
            return "", err_msg, -3  # Indicate other error with -3 