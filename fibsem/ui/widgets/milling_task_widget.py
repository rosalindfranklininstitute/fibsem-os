from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget


# TODO: be able to sync the position of different tasks

class FibsemMillingTaskWidget(QWidget):
    """Widget for selecting and configuring milling tasks.

    Contains a list widget for task selection and a configuration widget
    for editing the selected task's settings.
    """

    task_config_updated = pyqtSignal(str, FibsemMillingTaskConfig)  # task_name, config
    task_selection_changed = pyqtSignal(str)  # task_name

    def __init__(
        self,
        microscope: FibsemMicroscope,
        task_configs: Optional[Dict[str, FibsemMillingTaskConfig]] = None,
        milling_enabled: bool = True,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the milling task widget.

        Args:
            microscope: FibsemMicroscope instance
            task_configs: Dictionary of task configurations with task names as keys
            milling_enabled: Whether to show milling controls
            parent: Parent widget
        """
        super().__init__(parent)
        self.microscope = microscope
        self._task_configs = task_configs or {}
        self._milling_enabled = milling_enabled
        self._current_task_name: Optional[str] = None

        self._setup_ui()
        self._connect_signals()

        # Load initial task configs
        if self._task_configs:
            self._populate_task_list()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Create splitter for resizable panes
        self.splitter = QSplitter()
        layout.addWidget(self.splitter)

        # Left pane: Task list
        self.task_list = QListWidget()
        self.task_list.setMaximumWidth(200)
        self.task_list.setMinimumWidth(100)
        self.splitter.addWidget(self.task_list)

        # Right pane: Task configuration widget
        self.config_widget = MillingTaskConfigWidget(
            microscope=self.microscope,
            milling_enabled=self._milling_enabled,
            parent=self,
        )
        self.splitter.addWidget(self.config_widget)

        # Set splitter proportions (1:3 ratio)
        self.splitter.setSizes([200, 600])

    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.task_list.currentItemChanged.connect(self._on_task_selection_changed)
        self.config_widget.settings_changed.connect(self._on_config_changed)

    def _populate_task_list(self):
        """Populate the task list with available task configurations."""
        self.task_list.clear()
        for task_name in self._task_configs.keys():
            item = QListWidgetItem(task_name)
            self.task_list.addItem(item)
            
        # Select the first task if available
        if self.task_list.count() > 0:
            self.task_list.setCurrentRow(0)

        # if only one task, hide the task_list
        self.task_list.setVisible(self.task_list.count() > 1)

    def _on_task_selection_changed(
        self, current: QListWidgetItem, _previous: QListWidgetItem
    ):
        """Handle task selection changes."""
        if current is None:
            self._current_task_name = None
            return

        task_name = current.text()
        self._current_task_name = task_name

        # Load the selected task configuration
        if task_name in self._task_configs:
            # Update background milling stages with other tasks
            self._update_background_milling_stages()

            # update config widget with selected task settings
            self.config_widget.update_from_settings(self._task_configs[task_name])
            # NOTE: selection seems to cause a double draw in stage editor widget

    def _on_config_changed(self, config: FibsemMillingTaskConfig):
        """Handle configuration changes in the sub-widget."""
        print(f"_on_config_changed called: current_task='{self._current_task_name}', config={config is not None}")
        if self._current_task_name:
            # Update the stored configuration
            self._task_configs[self._current_task_name] = config
            self.task_config_updated.emit(self._current_task_name, config)
            print(f"Task '{self._current_task_name}' configuration updated and signal emitted.")
        else:
            # When no current task is selected, emit update for all task configs
            # This handles the case where the lamella editor passes all configs
            print("No current task selected, emitting updates for all tasks")
            for task_name in self._task_configs.keys():
                self.task_config_updated.emit(task_name, config)
                print(f"Emitted update for task '{task_name}'")
                break  # Only emit once since all tasks share the same config in this case

    def set_task_configs(self, task_configs: Dict[str, FibsemMillingTaskConfig]):
        """Set the available task configurations.

        Args:
            task_configs: Dictionary of task configurations with task names as keys
        """
        self._task_configs = task_configs.copy()
        self._populate_task_list()

        # Clear current selection
        self.task_list.clearSelection()
        self._current_task_name = None

    def get_task_configs(self) -> Dict[str, FibsemMillingTaskConfig]:
        """Get all task configurations.

        Returns:
            Dictionary of task configurations with task names as keys
        """
        return self._task_configs.copy()

    def get_current_task_config(self) -> Optional[FibsemMillingTaskConfig]:
        """Get the currently selected task configuration.

        Returns:
            The currently selected task configuration, or None if no task is selected
        """
        if self._current_task_name and self._current_task_name in self._task_configs:
            return self._task_configs[self._current_task_name]
        return None

    def get_current_task_name(self) -> Optional[str]:
        """Get the name of the currently selected task.

        Returns:
            The name of the currently selected task, or None if no task is selected
        """
        return self._current_task_name

    def select_task(self, task_name: str):
        """Programmatically select a task by name.

        Args:
            task_name: Name of the task to select
        """
        if task_name not in self._task_configs:
            return

        # Find and select the item
        for i in range(self.task_list.count()):
            item = self.task_list.item(i)
            if item and item.text() == task_name:
                self.task_list.setCurrentItem(item)
                break

    def add_task_config(self, task_name: str, config: FibsemMillingTaskConfig):
        """Add a new task configuration.

        Args:
            task_name: Name of the task
            config: Task configuration
        """
        self._task_configs[task_name] = config
        self._populate_task_list()

    def remove_task_config(self, task_name: str):
        """Remove a task configuration.

        Args:
            task_name: Name of the task to remove
        """
        if task_name in self._task_configs:
            del self._task_configs[task_name]
            self._populate_task_list()

            # Clear selection if the removed task was selected
            if self._current_task_name == task_name:
                self.task_list.clearSelection()
                self._current_task_name = None

    def update_task_config(self, task_name: str, config: FibsemMillingTaskConfig):
        """Update an existing task configuration.

        Args:
            task_name: Name of the task to update
            config: New task configuration
        """
        if task_name in self._task_configs:
            self._task_configs[task_name] = config

            # If this is the currently selected task, update the widget
            if self._current_task_name == task_name:
                self.config_widget.update_from_settings(config)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in the configuration widget.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self.config_widget.set_show_advanced(show_advanced)

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self.config_widget.get_show_advanced()

    def _update_background_milling_stages(self):
        """Update background milling stages with stages from other tasks."""
        if not self._current_task_name:
            return

        background_stages = []
        for task_name, task_config in self._task_configs.items():
            if task_name != self._current_task_name:
                background_stages.extend(task_config.stages)

        self.config_widget.set_background_milling_stages(background_stages)


if __name__ == "__main__":
    import os
    from pathlib import Path

    import napari
    from PyQt5.QtWidgets import QTabWidget, QWidget

    from fibsem.applications.autolamella.structures import (
        AutoLamellaProtocol,
        Experiment,
    )

    from fibsem import utils
    from PyQt5.QtWidgets import QVBoxLayout

    viewer = napari.Viewer()
    main_widget = QTabWidget()

    # set tab to side
    qwidget = QWidget()
    icon1 = QIconifyIcon("material-symbols:experiment", color="white")
    main_widget.addTab(qwidget, icon1, "Experiment")  # type: ignore
    layout = QVBoxLayout()
    qwidget.setLayout(layout)
    qwidget.setContentsMargins(0, 0, 0, 0)
    layout.setContentsMargins(0, 0, 0, 0)

    microscope, settings = utils.setup_session()

    BASE_PATH = (
        "/home/patrick/github/autolamella/autolamella/log/AutoLamella-2025-05-28-17-22/"
    )
    EXPERIMENT_PATH = Path(os.path.join(BASE_PATH, "experiment.yaml"))
    PROTOCOL_PATH = Path(os.path.join(BASE_PATH, "protocol.yaml"))
    exp = Experiment.load(EXPERIMENT_PATH)
    protocol = AutoLamellaProtocol.load(PROTOCOL_PATH)

    task_configs = {}

    task_configs["mill_rough"] = FibsemMillingTaskConfig.from_stages(
        stages=exp.positions[0].milling_workflows["mill_rough"],  # type: ignore
    )
    task_configs["mill_polishing"] = FibsemMillingTaskConfig.from_stages(
        stages=exp.positions[0].milling_workflows["mill_polishing"],  # type: ignore
    )
    # task_configs["stress-relief"] = FibsemMillingTaskConfig.from_stages(
    #     stages=exp.positions[1].milling_workflows["microexpansion"],  # type: ignore
    # )

    task_widget = FibsemMillingTaskWidget(
        microscope=microscope, task_configs=task_configs, milling_enabled=True
    )
    layout.addWidget(task_widget)

    # Connect to settings change signal
    def on_task_config_changed(task_config: FibsemMillingTaskConfig):
        print(f"Task Config changed: {utils.current_timestamp_v3(timeonly=False)}")
        print(f"  name: {task_config.name}")
        print(f"  field_of_view: {task_config.field_of_view}")
        print(f"  alignment: {task_config.alignment}")
        print(f"  acquisition: {task_config.acquisition}")
        print(f"  stages: {len(task_config.stages)} stages")
        for stage in task_config.stages:
            print(f"    Stage Name: {stage.name}")
            print(f"    Milling: {stage.milling}")
            print(f"    Pattern: {stage.pattern}")
            print(f"    Strategy: {stage.strategy.config}")
            print("---------------------" * 3)

    # config_widget.settings_changed.connect(on_task_config_changed)
    main_widget.setWindowTitle("MillingTaskConfig Widget Test")

    viewer.window.add_dock_widget(main_widget, add_vertical_stretch=False, area="right")

    napari.run()
