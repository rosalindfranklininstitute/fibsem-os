import os
from typing import Optional, Type

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QMenuBar,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from fibsem.applications.autolamella.config import BASE_PATH
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.workflows.tasks import TASK_REGISTRY
from fibsem.ui.widgets.autolamella_task_config_widget import AutoLamellaTaskConfigWidget
from fibsem.ui.widgets.autolamella_workflow_config_widget import (
    AutoLamellaWorkflowConfigWidget,
)

TASK_PROTOCOL_PATH = os.path.join(
    BASE_PATH, "protocol", "autolamella-task-protocol.yaml"
)
DEFAULT_PROTOCOL = AutoLamellaTaskProtocol.load(TASK_PROTOCOL_PATH)

# TODO: support multiple of the same task-type -> migrate to task-type rather than name

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
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.protocol = protocol or AutoLamellaTaskProtocol()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create menu bar
        self._create_menu_bar(main_layout)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget for scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)
        content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setContentsMargins(0, 0, 0, 0)

        # Add the workflow config widget
        self.workflow_widget = AutoLamellaWorkflowConfigWidget(
            workflow_config=self.protocol.workflow_config, parent=self
        )
        self.workflow_collapsible = QCollapsible("Workflow Configuration", self)
        self.workflow_collapsible.addWidget(self.workflow_widget)
        self.workflow_collapsible.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.workflow_collapsible)  # type: ignore

        # Add the task config widget
        self.task_config_widget = AutoLamellaTaskConfigWidget()
        self.task_collapsible = QCollapsible("Task Configuration", self)
        self.task_collapsible.addWidget(self.task_config_widget)
        self.task_collapsible.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.task_collapsible)  # type: ignore
        content_layout.addStretch()

        # Set content widget in scroll area and add to main layout
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    def _create_menu_bar(self, main_layout: QVBoxLayout):
        """Create and configure the menu bar."""
        menu_bar = QMenuBar()
        main_layout.addWidget(menu_bar)

        # Create File menu
        file_menu = menu_bar.addMenu("File")
        if file_menu is None:
            return

        # Add Save Protocol action
        save_action = file_menu.addAction("Save Protocol")
        if save_action:
            save_action.triggered.connect(self._save_protocol)

        # Add Load Protocol action
        load_action = file_menu.addAction("Load Protocol")
        if load_action:
            load_action.triggered.connect(self._load_protocol)

    def _save_protocol(self):
        """Save the current protocol to a YAML file."""
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                caption="Save Protocol",
                directory=os.path.join(
                    os.path.dirname(TASK_PROTOCOL_PATH),
                    "autolamella-protocol.yaml"),
                filter="YAML Files (*.yaml *.yml);;All Files (*)",
            )
            if file_path == "":
                return

            # Get current protocol
            protocol = self.get_protocol()
            protocol.save(file_path)

            QMessageBox.information(
                self, "Success", f"Protocol saved successfully to:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save protocol:\n{str(e)}")

    def _load_protocol(self):
        """Load a protocol from a YAML file."""
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Load Protocol",
                directory=os.path.dirname(TASK_PROTOCOL_PATH),
                filter="YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if file_path == "":
                return

            # Load protocol from YAML file
            protocol = AutoLamellaTaskProtocol.load(file_path)
            self.set_protocol(protocol)

            QMessageBox.information(
                self, "Success", f"Protocol loaded successfully from:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load protocol:\n{str(e)}")

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
        print(
            f"Protocol widget: Task config changed - {task_config.__class__.__name__}"
        )
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
        widget, name="AutoLamella Task Protocol", add_vertical_stretch=False
    )

    napari.run()
