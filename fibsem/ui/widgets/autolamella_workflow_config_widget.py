from typing import Dict, List, Optional
from copy import deepcopy

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.workflows.tasks import TASK_REGISTRY
from fibsem.ui.stylesheets import (
    GREEN_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
    BLUE_PUSHBUTTON_STYLE,
)


class TaskDescriptionWidget(QFrame):
    """Widget for displaying and editing a single task description."""

    task_changed = pyqtSignal(AutoLamellaTaskDescription)
    remove_requested = pyqtSignal(AutoLamellaTaskDescription)
    move_up_requested = pyqtSignal(AutoLamellaTaskDescription)
    move_down_requested = pyqtSignal(AutoLamellaTaskDescription)
    task_selected = pyqtSignal(AutoLamellaTaskDescription)

    def __init__(
        self, task_desc: AutoLamellaTaskDescription, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.task_desc = task_desc
        self.selected = False
        self._setup_ui()
        self._connect_signals()
        self._update_from_task()

        # Make selectable frame
        self.setFrameStyle(QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # self.setContentsMargins(0, 0, 0, 0)

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Task name (display only)
        self.name_label = QLabel()
        # remove outline
        self.name_label.setStyleSheet("QLabel { border: none; }")
        layout.addWidget(self.name_label)

        # Required checkbox
        self.required_checkbox = QCheckBox("Required")
        layout.addWidget(self.required_checkbox)

        # Supervise checkbox
        self.supervise_checkbox = QCheckBox("Supervise")
        layout.addWidget(self.supervise_checkbox)

        # Move up button
        self.move_up_button = QPushButton("↑")
        self.move_up_button.setMaximumWidth(30)
        self.move_up_button.setToolTip("Move up")
        self.move_up_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        layout.addWidget(self.move_up_button)

        # Move down button
        self.move_down_button = QPushButton("↓")
        self.move_down_button.setMaximumWidth(30)
        self.move_down_button.setToolTip("Move down")
        self.move_down_button.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        layout.addWidget(self.move_down_button)

        # Remove button
        self.remove_button = QPushButton("✗")
        self.remove_button.setMaximumWidth(30)
        self.remove_button.setToolTip("Remove task")
        self.remove_button.setStyleSheet(RED_PUSHBUTTON_STYLE)
        layout.addWidget(self.remove_button)

    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.required_checkbox.toggled.connect(self._on_required_changed)
        self.supervise_checkbox.toggled.connect(self._on_supervise_changed)
        self.move_up_button.clicked.connect(
            lambda: self.move_up_requested.emit(self.task_desc)
        )
        self.move_down_button.clicked.connect(
            lambda: self.move_down_requested.emit(self.task_desc)
        )
        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self.task_desc)
        )

    def _update_from_task(self):
        """Update widget from task description."""
        # Get display name from task registry if available
        display_name = self.task_desc.name
        if self.task_desc.name in TASK_REGISTRY:
            task_class = TASK_REGISTRY[self.task_desc.name]
            if hasattr(task_class.config_cls, "display_name"):
                display_name = task_class.config_cls.display_name

        self.name_label.setText(display_name)
        self.required_checkbox.setChecked(self.task_desc.required)
        self.supervise_checkbox.setChecked(self.task_desc.supervise)

    def _on_required_changed(self, checked: bool):
        """Handle required checkbox change."""
        self.task_desc.required = checked
        self.task_changed.emit(self.task_desc)

    def _on_supervise_changed(self, checked: bool):
        """Handle supervise checkbox change."""
        self.task_desc.supervise = checked
        self.task_changed.emit(self.task_desc)

    def set_move_buttons_enabled(self, can_move_up: bool, can_move_down: bool):
        """Enable/disable move buttons based on position in list."""
        self.move_up_button.setEnabled(can_move_up)
        self.move_down_button.setEnabled(can_move_down)

    def set_selected(self, selected: bool):
        """Set the selected state and update visual appearance."""
        self.selected = selected
        if selected:
            # Napari-style selection with dark theme
            self.setStyleSheet("""
                QFrame { 
                    background-color: rgba(0, 122, 204, 0.3); 
                    border: 2px solid #007ACC; 
                    border-radius: 4px;
                }
            """)
        else:
            # Napari-style unselected with dark theme
            self.setStyleSheet("""
                QFrame { 
                    background-color: rgba(40, 40, 40, 0.5); 
                    border: none;
                    border-radius: 4px;
                }
                QFrame:hover {
                    background-color: rgba(60, 60, 60, 0.7);
                    border: 1px solid #505050;
                }
            """)

    def mousePressEvent(self, event):
        """Handle mouse press for selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.task_selected.emit(self.task_desc)
        super().mousePressEvent(event)


class AutoLamellaWorkflowConfigWidget(QWidget):
    """Widget for configuring AutoLamella workflow tasks with drag-drop reordering."""

    workflow_changed = pyqtSignal(AutoLamellaWorkflowConfig)
    task_selected = pyqtSignal(
        AutoLamellaTaskDescription
    )  # Emitted when a task is selected

    def __init__(
        self,
        workflow_config: Optional[AutoLamellaWorkflowConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.workflow_config = workflow_config or AutoLamellaWorkflowConfig()
        self._task_widgets: Dict[str, TaskDescriptionWidget] = {}
        self._selected_task: Optional[str] = None

        self._setup_ui()
        self._connect_signals()
        self._update_from_config()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Add task controls
        add_task_layout = QHBoxLayout()
        self.task_combo = QComboBox()
        self._populate_task_combo()
        add_task_layout.addWidget(self.task_combo)

        self.add_task_button = QPushButton("Add Task")
        self.add_task_button.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        add_task_layout.addWidget(self.add_task_button)

        layout.addLayout(add_task_layout)

        # Task list container
        self.task_list_widget = QWidget()
        self.task_list_layout = QVBoxLayout()
        self.task_list_layout.setContentsMargins(0, 0, 0, 0)
        self.task_list_widget.setContentsMargins(0, 0, 0, 0)
        self.task_list_widget.setLayout(self.task_list_layout)
        layout.addWidget(self.task_list_widget)

        # Help text
        help_label = QLabel(
            "Use ↑/↓ buttons to reorder tasks. Use checkboxes to configure task properties."
        )
        help_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(help_label)

    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.add_task_button.clicked.connect(self._on_add_task)

    def _populate_task_combo(self):
        """Populate the task combo box with available tasks."""
        self.task_combo.clear()
        self.task_combo.addItem("Select a task...", "")

        for task_name, task_class in TASK_REGISTRY.items():
            display_name = task_name
            if hasattr(task_class.config_cls, "display_name"):
                display_name = f"{task_class.config_cls.display_name}"
            self.task_combo.addItem(display_name, task_name)

    def _update_from_config(self):
        """Update widget from workflow config."""
        # Clear existing task widgets and selection
        self._clear_task_widgets()
        self._selected_task = None

        # Add task widgets for each task in order
        for task_desc in self.workflow_config.tasks:
            self._add_task_widget(task_desc)

    def _clear_task_widgets(self):
        """Remove all task widgets."""
        for widget in self._task_widgets.values():
            widget.setParent(None)
        self._task_widgets.clear()

    def _add_task_widget(self, task_desc: AutoLamellaTaskDescription):
        """Add a task widget for the given task description."""
        if task_desc.name in self._task_widgets:
            return  # Already exists

        widget = TaskDescriptionWidget(task_desc, self)
        widget.task_changed.connect(self._on_task_changed)
        widget.remove_requested.connect(self._on_remove_task)
        widget.move_up_requested.connect(self._on_move_up_task)
        widget.move_down_requested.connect(self._on_move_down_task)
        widget.task_selected.connect(self._on_task_selected)

        self._task_widgets[task_desc.name] = widget
        self.task_list_layout.addWidget(widget)
        self._update_move_buttons()

    def _on_name_changed(self, text: str):
        """Handle workflow name change."""
        self.workflow_config.name = text
        self.workflow_changed.emit(self.workflow_config)

    def _on_description_changed(self, text: str):
        """Handle workflow description change."""
        self.workflow_config.description = text
        self.workflow_changed.emit(self.workflow_config)

    def _on_add_task(self):
        """Handle add task button click."""
        task_name = self.task_combo.currentData()
        if not task_name:
            return

        # Check if task already exists
        if any(task.name == task_name for task in self.workflow_config.tasks):
            QMessageBox.warning(
                self, "Task Exists", f"Task '{task_name}' is already in the workflow."
            )
            return

        # Create new task description
        task_desc = AutoLamellaTaskDescription(
            name=task_name,
            supervise=True,  # Default to requiring supervision
            required=True,  # Default to required
            requires=[],
        )

        # Add to workflow config
        self.workflow_config.tasks.append(task_desc)

        # Add widget
        self._add_task_widget(task_desc)

        # Reset combo selection
        self.task_combo.setCurrentIndex(0)

        self.workflow_changed.emit(self.workflow_config)

    def _on_remove_task(self, task_desc: AutoLamellaTaskDescription):
        """Handle task removal request."""
        # Get display name for the confirmation dialog
        display_name = self._get_display_name(task_desc.name)

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Remove Task",
            f"Are you sure you want to remove the task '{display_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,  # Default to No for safety
        )

        # Only proceed if user confirms
        if reply != QMessageBox.Yes:
            return

        # Remove from workflow config
        self.workflow_config.tasks = [
            t for t in self.workflow_config.tasks if t.name != task_desc.name
        ]

        # Remove widget
        if task_desc.name in self._task_widgets:
            widget = self._task_widgets[task_desc.name]
            widget.setParent(None)
            del self._task_widgets[task_desc.name]

        # Clear selection if removed task was selected
        if self._selected_task == task_desc.name:
            self._selected_task = None

        self._update_move_buttons()
        self.workflow_changed.emit(self.workflow_config)

    def _on_task_changed(self, task_desc: AutoLamellaTaskDescription):
        """Handle task description change."""
        # Update the task in workflow config
        for i, task in enumerate(self.workflow_config.tasks):
            if task.name == task_desc.name:
                self.workflow_config.tasks[i] = task_desc
                break

        self.workflow_changed.emit(self.workflow_config)

    def _on_task_selected(self, task_desc: AutoLamellaTaskDescription):
        """Handle task selection."""
        # Don't reselect if already selected
        if self._selected_task == task_desc.name:
            return

        # Update selection state
        self._selected_task = task_desc.name

        # Update visual state of all widgets
        for task_name, widget in self._task_widgets.items():
            widget.set_selected(task_name == self._selected_task)

        # Emit signal for external listeners
        self.task_selected.emit(task_desc)

        # Print task content for debugging
        print(f"Selected task: {task_desc.name}")
        print(f"  Display name: {self._get_display_name(task_desc.name)}")
        print(f"  Required: {task_desc.required}")
        print(f"  Supervise: {task_desc.supervise}")
        print(f"  Requires: {task_desc.requires}")
        print("=" * 50)

    def _get_display_name(self, task_name: str) -> str:
        """Get the display name for a task."""
        if task_name in TASK_REGISTRY:
            task_class = TASK_REGISTRY[task_name]
            if hasattr(task_class.config_cls, "display_name"):
                return task_class.config_cls.display_name
        return task_name

    def _on_move_up_task(self, task_desc: AutoLamellaTaskDescription):
        """Handle move up request for a task."""
        task_names = [task.name for task in self.workflow_config.tasks]
        current_index = task_names.index(task_desc.name)

        if current_index > 0:
            # Swap with previous task
            task_names[current_index], task_names[current_index - 1] = (
                task_names[current_index - 1],
                task_names[current_index],
            )
            self.reorder_tasks(task_names)

    def _on_move_down_task(self, task_desc: AutoLamellaTaskDescription):
        """Handle move down request for a task."""
        task_names = [task.name for task in self.workflow_config.tasks]
        current_index = task_names.index(task_desc.name)

        if current_index < len(task_names) - 1:
            # Swap with next task
            task_names[current_index], task_names[current_index + 1] = (
                task_names[current_index + 1],
                task_names[current_index],
            )
            self.reorder_tasks(task_names)

    def _update_move_buttons(self):
        """Update the enabled state of move buttons for all task widgets."""
        task_count = len(self.workflow_config.tasks)

        for i, task in enumerate(self.workflow_config.tasks):
            if task.name in self._task_widgets:
                widget = self._task_widgets[task.name]
                can_move_up = i > 0
                can_move_down = i < task_count - 1
                widget.set_move_buttons_enabled(can_move_up, can_move_down)

    def reorder_tasks(self, task_names: List[str]):
        """Reorder tasks based on provided task names list."""
        # Create new task list in the specified order
        new_tasks = []
        existing_tasks = {task.name: task for task in self.workflow_config.tasks}

        for task_name in task_names:
            if task_name in existing_tasks:
                new_tasks.append(existing_tasks[task_name])

        self.workflow_config.tasks = new_tasks

        # Update widget order
        self._update_widget_order(task_names)

        # Update move button states
        self._update_move_buttons()

        self.workflow_changed.emit(self.workflow_config)

    def _update_widget_order(self, task_names: List[str]):
        """Update the visual order of task widgets."""
        # Remove all widgets from layout
        for widget in self._task_widgets.values():
            self.task_list_layout.removeWidget(widget)

        # Re-add widgets in new order
        for task_name in task_names:
            if task_name in self._task_widgets:
                self.task_list_layout.addWidget(self._task_widgets[task_name])

    def get_workflow_config(self) -> AutoLamellaWorkflowConfig:
        """Get the current workflow configuration."""
        return deepcopy(self.workflow_config)

    def set_workflow_config(self, workflow_config: AutoLamellaWorkflowConfig):
        """Set the workflow configuration."""
        self.workflow_config = deepcopy(workflow_config)
        self._update_from_config()


if __name__ == "__main__":
    # import sys
    # from PyQt5.QtWidgets import QApplication

    # app = QApplication(sys.argv)

    # Create test workflow config
    import napari
    from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription

    test_config = AutoLamellaWorkflowConfig(
        name="Test Workflow",
        description="A test workflow for demonstration",
        tasks=[
            AutoLamellaTaskDescription(
                name="SETUP_LAMELLA", supervise=True, required=True
            ),
            AutoLamellaTaskDescription(
                name="MILL_ROUGH", supervise=False, required=True
            ),
            AutoLamellaTaskDescription(
                name="MILL_POLISHING", supervise=True, required=False
            ),
        ],
    )

    def print_workflow_config(config: AutoLamellaWorkflowConfig):
        print(f"Workflow Name: {config.name}")
        print(f"Description: {config.description}")
        print("Tasks:")
        for task in config.tasks:
            print(
                f" - {task.name} (Supervise: {task.supervise}, Required: {task.required})"
            )

    viewer = napari.Viewer()

    widget = AutoLamellaWorkflowConfigWidget(test_config)
    widget.workflow_changed.connect(lambda config: print_workflow_config(config))
    widget.task_selected.connect(
        lambda task: print(f"Task selected signal: {task.name}")
    )

    # widget.setWindowTitle("AutoLamella Workflow Config Widget Test")
    # widget.resize(600, 400)
    # widget.show()

    viewer.window.add_dock_widget(
        widget, name="AutoLamella Workflow Config", add_vertical_stretch=True
    )

    napari.run()
    # sys.exit(app.exec_())
