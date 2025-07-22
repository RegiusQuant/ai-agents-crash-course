import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional, Type

# Configure module logger
tool_logger = logging.getLogger(__name__)
tool_logger.addHandler(logging.NullHandler())

type_aliases: Dict[str, Type[Any]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
}


class FunctionSignature:
    """
    Models a function's name, documentation, and parameters for dynamic invocation.
    """

    def __init__(self, func: Callable) -> None:
        sig = inspect.signature(func)
        self.name: str = func.__name__
        self.description: Optional[str] = (func.__doc__ or "").strip() or None
        self.parameters: Dict[str, Dict[str, Any]] = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            info: Dict[str, Any] = {}
            if param.annotation is not inspect._empty:
                annot = param.annotation
                info['type'] = annot.__name__ if hasattr(
                    annot, '__name__') else str(annot)
            if param.default is not inspect._empty:
                info['default'] = param.default
            self.parameters[param_name] = info

        self.return_type: Optional[str] = None
        if sig.return_annotation is not inspect._empty:
            ret = sig.return_annotation
            self.return_type = ret.__name__ if hasattr(
                ret, '__name__') else str(ret)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ArgumentValidator:
    """
    Validates and coerces arguments based on a FunctionSignature.
    """

    def __init__(self, type_map: Optional[Dict[str, Type[Any]]] = None) -> None:
        self._type_map = type_map or type_aliases

    def validate(self, args: Dict[str, Any], signature: FunctionSignature) -> Dict[str, Any]:
        validated: Dict[str, Any] = {}
        for name, meta in signature.parameters.items():
            if name in args:
                val = args[name]
                expected = meta.get('type')
                if expected and expected in self._type_map:
                    target_type = self._type_map[expected]
                    if not isinstance(val, target_type):
                        try:
                            val = target_type(val)
                        except Exception as e:
                            tool_logger.error(
                                "Failed to convert argument '%s' to %s: %s",
                                name, expected, e
                            )
                            raise TypeError(
                                f"Argument '{name}' must be of type {expected}")
                validated[name] = val
            else:
                if 'default' in meta:
                    validated[name] = meta['default']
                else:
                    raise KeyError(f"Missing required argument: '{name}'")
        return validated


class Tool:
    """
    Wraps a callable for standardized signature introspection and invocation.
    """

    def __init__(
        self,
        function: Callable,
        signature: FunctionSignature,
        validator: ArgumentValidator
    ) -> None:
        self._fn = function
        self.signature = signature
        self._validator = validator

    def __call__(self, **kwargs: Any) -> Any:
        args = self._validator.validate(kwargs, self.signature)
        tool_logger.debug(
            "Invoking '%s' with arguments: %s",
            self.signature.name, args
        )
        return self._fn(**args)

    def info(self) -> str:
        """Returns the JSON representation of this tool's signature."""
        return self.signature.to_json()


def tool(func: Callable) -> Tool:
    """
    Decorator that converts a function into a Tool with introspected signature.
    """
    sig = FunctionSignature(func)
    validator = ArgumentValidator()
    return Tool(function=func, signature=sig, validator=validator)