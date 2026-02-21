from logger import logger
class DetachHook:
    def __init__(self, hooks = []) -> None:
        self.hooks = hooks

    def detach_hooks(self) -> None:
        for hook in self.hooks:
            try:
                hook.remove()
                # logger.info("Hook detached successfully", context="DetachHook.detach_hooks")
            except Exception as e:
                logger.error(f"Failed to detach hook: {str(e)}", context="DetachHook.detach_hooks")
                