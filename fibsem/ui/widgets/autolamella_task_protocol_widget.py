import os
from typing import Optional, Type

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
    AutoLamellaTaskConfig,
    AutoLamellaTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks import TASK_REGISTRY
from fibsem.ui.widgets.autolamella_workflow_config_widget import AutoLamellaWorkflowConfigWidget
from fibsem.ui.widgets.autolamella_task_config_widget import AutoLamellaTaskConfigWidget
from fibsem.applications.autolamella.config import BASE_PATH


TASK_PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol", "autolamella-task-protocol.yaml")
DEFAULT_PROTOCOL = AutoLamellaTaskProtocol.load(TASK_PROTOCOL_PATH)


class AutoLamellaTaskProtocolWidget(QWidget):
    """Widget for configuring AutoLamella task protocols including workflow and task configurations."""
    
    # Signals
    protocol_changed = pyqtSignal(AutoLamellaTaskProtocol)
    workflow_changed = pyqtSignal(AutoLamellaWorkflowConfig)
    task_selected = pyqtSignal(AutoLamellaTaskDescription)
    task_config_changed = pyqtSignal(AutoLamellaTaskConfig)
    
    def __init__(
        self, 
        protocol: Optional[AutoLamellaTaskProtocol] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.protocol = protocol or AutoLamellaTaskProtocol()
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Create and configure all UI elements."""
        # Main vertical splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)
        
        # Add the workflow config widget
        self.workflow_widget = AutoLamellaWorkflowConfigWidget(
            workflow_config=self.protocol.workflow_config,
            parent=self
        )
        main_splitter.addWidget(self.workflow_widget)
        
        # Add the task config widget
        self.task_config_widget = AutoLamellaTaskConfigWidget()
        main_splitter.addWidget(self.task_config_widget)
        
        # Set initial splitter proportions (60% workflow, 40% task config)
        main_splitter.setSizes([600, 400])
    
    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        # Connect workflow widget signals to our signals
        self.workflow_widget.workflow_changed.connect(self._on_workflow_changed)
        self.workflow_widget.task_selected.connect(self._on_task_selected)
        # Connect the task config widget's signal directly
        self.task_config_widget.config_changed.connect(self._on_task_config_changed)
    
    def _on_workflow_changed(self, workflow_config: AutoLamellaWorkflowConfig):
        """Handle workflow configuration changes."""
        self.protocol.workflow_config = workflow_config
        print(f"Protocol widget: Workflow changed - {workflow_config.name}")
        self.workflow_changed.emit(workflow_config)
        self.protocol_changed.emit(self.protocol)
    
    def _on_task_selected(self, task_desc: AutoLamellaTaskDescription):
        """Handle task selection."""
        print(f"Protocol widget: Task selected - {task_desc.name}")
        
        # Show the task configuration for the selected task
        self._show_task_config(task_desc)
        
        self.task_selected.emit(task_desc)
    
    def _on_task_config_changed(self, task_config: AutoLamellaTaskConfig):
        """Handle task configuration parameter changes."""
        # Store the task config in the protocol's task_config dictionary
        task_name = task_config.task_name
        self.protocol.task_config[task_name] = task_config
        print(f"Protocol widget: Task config changed - {task_config.__class__.__name__}")
        self.task_config_changed.emit(task_config)
        self.protocol_changed.emit(self.protocol)
    
    def _show_task_config(self, task_desc: AutoLamellaTaskDescription):
        """Show the task configuration widget for the selected task."""
        if task_desc.name not in TASK_REGISTRY:
            # Clear task config widget if task not found
            self.task_config_widget.set_task_config(None)
            return
            
        # Check if we already have a task config for this task
        if task_desc.name in self.protocol.task_config:
            # Use existing task config
            task_config = self.protocol.task_config[task_desc.name]
        else:
            # Create new default config instance
            task_config = DEFAULT_PROTOCOL.task_config.get(task_desc.name)
            if task_config is None:
                raise ValueError(f"No default config found for task: {task_desc.name}")
            # TODO: make sure we have valid default milling for each new config
            # task_class = TASK_REGISTRY[task_desc.name]
            # task_config_cls: Type[AutoLamellaTaskConfig] = task_class.config_cls  # type: ignore
            # task_config = task_config_cls()
            # Store it in the protocol
            self.protocol.task_config[task_desc.name] = task_config
        
        # Set the task config in the widget
        self.task_config_widget.set_task_config(task_config)
    
    def get_protocol(self) -> AutoLamellaTaskProtocol:
        """Get the current protocol configuration."""
        # Update protocol with latest workflow config
        self.protocol.workflow_config = self.workflow_widget.get_workflow_config()
        return self.protocol
    
    def set_protocol(self, protocol: AutoLamellaTaskProtocol):
        """Set the protocol configuration."""
        self.protocol = protocol
        self.workflow_widget.set_workflow_config(protocol.workflow_config)
        
    def get_workflow_config(self) -> AutoLamellaWorkflowConfig:
        """Get the current workflow configuration."""
        return self.workflow_widget.get_workflow_config()
    
    def set_workflow_config(self, workflow_config: AutoLamellaWorkflowConfig):
        """Set the workflow configuration."""
        self.protocol.workflow_config = workflow_config
        self.workflow_widget.set_workflow_config(workflow_config)


if __name__ == "__main__":
    import napari
    from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription
    
    # # Create test protocol
    # test_workflow_config = AutoLamellaWorkflowConfig(
    #     name="Test Workflow",
    #     description="A test workflow for demonstration",
    #     tasks=[
    #         AutoLamellaTaskDescription(
    #             name="SETUP_LAMELLA", supervise=True, required=True
    #         ),
    #         AutoLamellaTaskDescription(
    #             name="MILL_ROUGH", supervise=False, required=True
    #         ),
    #         AutoLamellaTaskDescription(
    #             name="MILL_POLISHING", supervise=True, required=False
    #         ),
    #     ],
    # )
    
    # test_protocol = AutoLamellaTaskProtocol(
    #     name="Test Protocol",
    #     description="A test protocol for demonstration", 
    #     version="1.0",
    #     workflow_config=test_workflow_config,
    #     task_config={}
    # )
    
    def print_workflow_config(config: AutoLamellaWorkflowConfig):
        print(f"Protocol Name: {config.name}")
        print(f"Description: {config.description}")
        print("Tasks:")
        for task in config.tasks:
            print(f" - {task.name} (Supervise: {task.supervise}, Required: {task.required})")
    
    viewer = napari.Viewer()
    
    widget = AutoLamellaTaskProtocolWidget(DEFAULT_PROTOCOL)
    widget.protocol_changed.connect(lambda protocol: print(f"Protocol changed: {protocol.name}"))
    widget.workflow_changed.connect(lambda config: print_workflow_config(config))
    widget.task_selected.connect(lambda task: print(f"Protocol signal - Task selected: {task.name}"))
    widget.task_config_changed.connect(lambda config: print(f"Protocol signal - Task config changed: {config}"))
    
    viewer.window.add_dock_widget(
        widget, name="AutoLamella Task Protocol", add_vertical_stretch=True
    )
    
    napari.run()