import os
import json
from datetime import datetime

class TraceLogger:
    """Observability system that splits and writes agent reasoning, tool calls, and system events to clean log streams."""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Categorized log file paths
        self.master_log_path = os.path.join(self.log_dir, "adep_master.log")
        self.system_log_path = os.path.join(self.log_dir, "system.log")
        self.thinking_log_path = os.path.join(self.log_dir, "thinking.log")
        self.tool_log_path = os.path.join(self.log_dir, "tool_calls.log")

        # Initialize/touch log files to guarantee they exist immediately upon setup
        for log_path in [self.master_log_path, self.system_log_path, self.thinking_log_path, self.tool_log_path]:
            with open(log_path, "a", encoding="utf-8"):
                pass


    def _write_log(self, filepath, message, print_to_stdout=True):
        """Append timestamped message to target category log and the master log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        
        # Write to category-specific file
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(formatted_msg)
            
        # Append to chronological master file
        with open(self.master_log_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg)
            
        # Print cleanly to stdout only if enabled
        if print_to_stdout:
            print(formatted_msg.strip())

    def log_system(self, stage: str, message: str):
        """Log pipeline control flow and user decisions."""
        self._write_log(self.system_log_path, f"[SYSTEM-{stage.upper()}] {message}")

    def log_thinking(self, agent_name: str, thought: str):
        """Log agent thoughts and reasoning (File only, hide from console)."""
        self._write_log(self.thinking_log_path, f"[{agent_name.upper()}] Thinking: {thought}", print_to_stdout=False)

    def log_tool_call(self, agent_name: str, tool_name: str, arguments: dict):
        """Log what tool was invoked and with what arguments."""
        clean_args = dict(arguments)
        if "code" in clean_args:
            # Format multi-line code cleanly in log
            self._write_log(self.tool_log_path, f"[{agent_name.upper()}] CALLING TOOL: {tool_name} with code:\n{clean_args['code']}\n")
        else:
            self._write_log(self.tool_log_path, f"[{agent_name.upper()}] CALLING TOOL: {tool_name}({json.dumps(clean_args)})")

    def log_tool_output(self, agent_name: str, tool_name: str, output: str):
        """Log the output result of a tool run."""
        divider = "-" * 60
        self._write_log(self.tool_log_path, f"[{agent_name.upper()}] TOOL RESULT: {tool_name} returned:\n{output}\n{divider}")

    def log_adk_event(self, default_agent_name: str, event):
        """
        Parses Google ADK event objects dynamically, routing thinking, 
        tool calls, and tool outputs directly to the correct files.
        """
        author = event.author or default_agent_name
        if event.content and event.content.parts:
            for part in event.content.parts:
                # 1. Capture text thoughts
                if part.text:
                    self.log_thinking(author, part.text)
                
                # 2. Capture tool calls
                if part.function_call:
                    tool_name = part.function_call.name
                    args = dict(part.function_call.args) if part.function_call.args else {}
                    self.log_tool_call(author, tool_name, args)
                    
                # 3. Capture tool output
                if part.function_response:
                    tool_name = part.function_response.name
                    res = part.function_response.response
                    resp_content = res.get("result", str(res)) if isinstance(res, dict) else str(res)
                    self.log_tool_output(author, tool_name, resp_content)
