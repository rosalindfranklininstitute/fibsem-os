from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget


class AutoLamellaLamellaEditor(QWidget):
    """Widget for editing lamellae in an AutoLamella experiment.

    Contains a list widget for lamella selection and a milling task widget
    for editing the selected lamella's milling tasks.
    """

    experiment_updated = pyqtSignal(Experiment)  # experiment
    lamella_selection_changed = pyqtSignal(str)  # lamella_petname

    def __init__(
        self,
        microscope: FibsemMicroscope,
        experiment: Optional[Experiment] = None,
        milling_enabled: bool = True,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the AutoLamella lamella editor widget.

        Args:
            microscope: FibsemMicroscope instance
            experiment: Experiment object containing lamellae
            milling_enabled: Whether to show milling controls
            parent: Parent widget
        """
        super().__init__(parent)
        self.microscope = microscope
        self._experiment = experiment
        self._milling_enabled = milling_enabled
        self._current_lamella: Optional[Lamella] = None
        self._current_task_name: Optional[str] = None

        self._setup_ui()
        self._connect_signals()

        # Load initial experiment
        if self._experiment:
            self._populate_lamella_list()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Create main splitter for resizable panes
        main_splitter = QSplitter()
        layout.addWidget(main_splitter)

        # Left pane: Vertical splitter for lamella and task lists
        left_splitter = QSplitter()
        left_splitter.setOrientation(2)  # Vertical orientation
        left_splitter.setMaximumWidth(200)
        left_splitter.setMinimumWidth(100)
        main_splitter.addWidget(left_splitter)

        # Top left: Lamella list
        self.lamella_list = QListWidget()
        left_splitter.addWidget(self.lamella_list)

        # Bottom left: Task list  
        self.task_list = QListWidget()
        left_splitter.addWidget(self.task_list)

        # Set equal sizes for lamella and task lists
        left_splitter.setSizes([100, 100])

        # Right pane: Milling task widget
        self.task_widget = FibsemMillingTaskWidget(
            microscope=self.microscope,
            task_configs={},
            milling_enabled=self._milling_enabled,
            parent=self,
        )
        main_splitter.addWidget(self.task_widget)

        # Set splitter proportions (1:3 ratio)
        main_splitter.setSizes([200, 600])

    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.lamella_list.currentItemChanged.connect(self._on_lamella_selection_changed)
        self.task_list.currentItemChanged.connect(self._on_task_selection_changed)
        self.task_widget.task_config_updated.connect(self._on_task_config_updated)

    def _populate_lamella_list(self):
        """Populate the lamella list with available lamellae."""
        self.lamella_list.clear()
        if not self._experiment:
            return

        for i, lamella in enumerate(self._experiment.positions):
            item = QListWidgetItem(str(lamella.name))
            self.lamella_list.addItem(item)

        # Select the first lamella if available
        if self.lamella_list.count() > 0:
            self.lamella_list.setCurrentRow(0)

    def _on_lamella_selection_changed(
        self, current: QListWidgetItem, _previous: QListWidgetItem
    ):
        """Handle lamella selection changes."""
        if current is None:
            self._current_lamella = None
            self.task_widget.set_task_configs({})
            return

        idx = self.lamella_list.row(current)
        if idx < 0 or idx >= len(self._experiment.positions):
            self._current_lamella = None
            self.task_widget.set_task_configs({})
            return

        self._current_lamella = self._experiment.positions[idx]
        self._current_task_name = None
        
        # Populate task list with available tasks for this lamella
        self._populate_task_list()

        self.lamella_selection_changed.emit(self._current_lamella.petname)

    def wj(
        self, lamella: Lamella
    ) -> Dict[str, FibsemMillingTaskConfig]:
        """Create task configs from lamella's task_config.milling."""
        task_configs = {}

        if lamella.task_config:
            for task_name, autolamella_task_config in lamella.task_config.items():
                if autolamella_task_config.milling:
                    # Merge all milling configs from this task
                    for milling_name, milling_config in autolamella_task_config.milling.items():
                        # Use task_name.milling_name as key to avoid conflicts
                        key = f"{task_name}.{milling_name}" if len(autolamella_task_config.milling) > 1 else task_name
                        task_configs[key] = milling_config

        return task_configs

    def _populate_task_list(self):
        """Populate the task list with available tasks for the current lamella."""
        self.task_list.clear()
        if not self._current_lamella or not self._current_lamella.task_config:
            return

        for task_name in self._current_lamella.task_config.keys():
            item = QListWidgetItem(str(task_name))
            self.task_list.addItem(item)

        # Select the first task if available
        if self.task_list.count() > 0:
            self.task_list.setCurrentRow(0)

    def _on_task_selection_changed(
        self, current: QListWidgetItem, _previous: QListWidgetItem
    ):
        """Handle task selection changes."""
        if current is None or not self._current_lamella:
            self._current_task_name = None
            self.task_widget.set_task_configs({})
            return

        self._current_task_name = current.text()
        
        # Get the selected task config and create milling task configs
        if self._current_task_name in self._current_lamella.task_config:
            autolamella_task = self._current_lamella.task_config[self._current_task_name]
            task_configs = autolamella_task.milling if autolamella_task.milling else {}
            print(f"Loading task '{self._current_task_name}' with {len(task_configs)} milling configs: {list(task_configs.keys())}")
            self.task_widget.set_task_configs(task_configs)
        else:
            print(f"Task '{self._current_task_name}' not found in lamella task_config")
            self.task_widget.set_task_configs({})

    def _on_task_config_updated(self, task_name: str, config: FibsemMillingTaskConfig):
        """Handle task configuration updates and sync back to experiment."""
        print(f"Task config update received: task_name='{task_name}', current_task='{self._current_task_name}'")
        if not self._current_lamella or not self._experiment or not self._current_task_name:
            print(f"Skipping update: lamella={self._current_lamella is not None}, experiment={self._experiment is not None}, task={self._current_task_name}")
            return

        # Ensure task_config exists and has the current task
        if not self._current_lamella.task_config:
            print(f"Warning: No task_config found for lamella '{self._current_lamella.name}'")
            return
            
        if self._current_task_name not in self._current_lamella.task_config:
            print(f"Warning: Task '{self._current_task_name}' not found in lamella '{self._current_lamella.name}'")
            return

        # Update the milling config in the current autolamella task
        autolamella_task = self._current_lamella.task_config[self._current_task_name]
        if not autolamella_task.milling:
            autolamella_task.milling = {}
        autolamella_task.milling[task_name] = config

        # Emit signal that experiment was updated
        self.experiment_updated.emit(self._experiment)
        print(f"Updated lamella '{self._current_lamella.name}' task '{self._current_task_name}' milling '{task_name}' config")

    def set_experiment(self, experiment: Experiment):
        """Set the experiment to be edited.

        Args:
            experiment: Experiment object containing lamellae
        """
        self._experiment = experiment
        self._populate_lamella_list()

        # Clear current selection
        self.lamella_list.clearSelection()
        self._current_lamella = None

    def get_experiment(self) -> Optional[Experiment]:
        """Get the current experiment.

        Returns:
            The current experiment object
        """
        return self._experiment

    def get_current_lamella(self) -> Optional[Lamella]:
        """Get the currently selected lamella.

        Returns:
            The currently selected lamella, or None if no lamella is selected
        """
        return self._current_lamella

    def get_current_lamella_name(self) -> Optional[str]:
        """Get the name of the currently selected lamella.

        Returns:
            The petname of the currently selected lamella, or None if no lamella is selected
        """
        return self._current_lamella.petname if self._current_lamella else None

    def select_lamella(self, lamella_petname: str):
        """Programmatically select a lamella by petname.

        Args:
            lamella_petname: Petname of the lamella to select
        """
        for i in range(self.lamella_list.count()):
            item = self.lamella_list.item(i)
            if item and item.text() == lamella_petname:
                self.lamella_list.setCurrentItem(item)
                break

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in the task widget.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self.task_widget.set_show_advanced(show_advanced)

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self.task_widget.get_show_advanced()


if __name__ == "__main__":
    import os
    from pathlib import Path

    import napari
    from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

    from fibsem import utils
    from fibsem.applications.autolamella.structures import (
        AutoLamellaProtocol,
        Experiment,
    )

    viewer = napari.Viewer()
    main_widget = QTabWidget()

    # Create tab widget
    qwidget = QWidget()
    main_widget.addTab(qwidget, "Lamella Editor")
    layout = QVBoxLayout()
    qwidget.setLayout(layout)
    qwidget.setContentsMargins(0, 0, 0, 0)
    layout.setContentsMargins(0, 0, 0, 0)

    # Setup microscope
    microscope, settings = utils.setup_session()

    # Load test experiment
    BASE_PATH = (
        # "/home/patrick/github/autolamella/autolamella/log/AutoLamella-2025-05-28-17-22/"
        "/home/patrick/github/fibsem/scripts/Test-Experiment/"
    )
    EXPERIMENT_PATH = Path(os.path.join(BASE_PATH, "experiment.yaml"))
    exp = Experiment.load(EXPERIMENT_PATH)

    # Create the lamella editor widget
    lamella_editor = AutoLamellaLamellaEditor(
        microscope=microscope, experiment=exp, milling_enabled=True
    )
    layout.addWidget(lamella_editor)

    # Connect to experiment update signal
    def on_experiment_updated(experiment: Experiment):
        print(f"Experiment updated: {utils.current_timestamp_v3(timeonly=False)}")
        print(f"  name: {experiment.name}")
        print(f"  positions: {len(experiment.positions)} lamellae")

    lamella_editor.experiment_updated.connect(on_experiment_updated)
    main_widget.setWindowTitle("AutoLamella Lamella Editor Test")

    viewer.window.add_dock_widget(
        main_widget, add_vertical_stretch=False, area="right"
    )

    napari.run()