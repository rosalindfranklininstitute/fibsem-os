from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QWidget,
)

from fibsem.constants import MICRO_TO_SI, SI_TO_MICRO
from fibsem.structures import FibsemMillingSettings, BeamType
from fibsem.utils import format_value

# GUI Configuration Constants
WIDGET_CONFIG = {
    "spot_size": {
        "range": (0.001, 1000),
        "decimals": 1,
        "step": 1.0,
        "default": 50.0,
        "suffix": " nm",
    },
    "rate": {
        "range": (0.001, 1000),
        "decimals": 3,
        "step": 0.001,
        "default": 3.0,
        "suffix": " mm³/nA/s",
    },
    "dwell_time": {
        "range": (0.001, 1000),
        "decimals": 2,
        "step": 0.01,
        "default": 1.0,
        "suffix": " μs",
    },
    "hfw": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
    },
    "spacing": {
        "range": (0.1, 10.0),
        "decimals": 1,
        "step": 0.1,
        "default": 1.0,
        "suffix": "",
    },
}

# Common milling currents (in Amperes)
COMMON_MILLING_CURRENTS = [
    10e-12,    # 10 pA
    20e-12,    # 20 pA
    50e-12,    # 50 pA
    100e-12,   # 100 pA
    200e-12,   # 200 pA
    500e-12,   # 500 pA
    1e-9,      # 1 nA
    2e-9,      # 2 nA
    5e-9,      # 5 nA
    10e-9,     # 10 nA
    20e-9,     # 20 nA
    50e-9,     # 50 nA
]

# Common milling voltages (in Volts)
COMMON_MILLING_VOLTAGES = [
    2e3,       # 2 keV
    5e3,       # 5 keV
    10e3,      # 10 keV
    15e3,      # 15 keV
    20e3,      # 20 keV
    25e3,      # 25 keV
    30e3,      # 30 keV
]

# Common patterning modes
PATTERNING_MODES = ["Serial", "Parallel", "Serpentine", "Raster"]

# Common application files
APPLICATION_FILES = ["Si", "Si-ccs", "Si-multipass", "GaAs", "Diamond", "Tungsten", "Copper", "Gold"]

# Common presets
PRESETS = [
    "30 keV; 2nA",
    "30 keV; 1nA", 
    "30 keV; 500pA",
    "30 keV; 200pA",
    "30 keV; 100pA",
    "30 keV; 50pA",
    "2 keV; 50pA",
    "5 keV; 100pA",
]


class MillingSettingsWidget(QWidget):
    settings_changed = pyqtSignal(FibsemMillingSettings)

    def __init__(self, parent=None, show_advanced=False):
        """Initialize the MillingSettings widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings (spot size, rate,
                          spacing, application file, preset)
        """
        super().__init__(parent)
        self._settings = FibsemMillingSettings()
        self._show_advanced = show_advanced
        self._setup_ui()
        self._connect_signals()
        self.update_from_settings(self._settings)
        # Initial visibility update
        self._update_advanced_visibility()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        row = 0

        # Milling Current (combobox with formatted values)
        layout.addWidget(QLabel("Milling Current"), row, 0)
        self.milling_current_combo = QComboBox()
        for current in COMMON_MILLING_CURRENTS:
            label = format_value(current, unit="A", precision=1)
            self.milling_current_combo.addItem(label, current)
        # Set default to 2 nA (2e-9)
        default_index = self.milling_current_combo.findData(2e-9)
        if default_index >= 0:
            self.milling_current_combo.setCurrentIndex(default_index)
        layout.addWidget(self.milling_current_combo, row, 1)
        row += 1

        # Dwell Time
        layout.addWidget(QLabel("Dwell Time"), row, 0)
        self.dwell_time_spinbox = QDoubleSpinBox()
        dwell_config = WIDGET_CONFIG["dwell_time"]
        self.dwell_time_spinbox.setRange(*dwell_config["range"])
        self.dwell_time_spinbox.setDecimals(dwell_config["decimals"])
        self.dwell_time_spinbox.setSingleStep(dwell_config["step"])
        self.dwell_time_spinbox.setValue(dwell_config["default"])
        self.dwell_time_spinbox.setSuffix(dwell_config["suffix"])
        layout.addWidget(self.dwell_time_spinbox, row, 1)
        row += 1

        # Field of View
        layout.addWidget(QLabel("Field of View"), row, 0)
        self.hfw_spinbox = QDoubleSpinBox()
        hfw_config = WIDGET_CONFIG["hfw"]
        self.hfw_spinbox.setRange(*hfw_config["range"])
        self.hfw_spinbox.setDecimals(hfw_config["decimals"])
        self.hfw_spinbox.setSingleStep(hfw_config["step"])
        self.hfw_spinbox.setValue(hfw_config["default"])
        self.hfw_spinbox.setSuffix(hfw_config["suffix"])
        layout.addWidget(self.hfw_spinbox, row, 1)
        row += 1

        # Milling Voltage (combobox with formatted values)
        layout.addWidget(QLabel("Milling Voltage"), row, 0)
        self.milling_voltage_combo = QComboBox()
        for voltage in COMMON_MILLING_VOLTAGES:
            label = format_value(voltage, unit="V", precision=0)
            self.milling_voltage_combo.addItem(label, voltage)
        # Set default to 30 keV (30e3)
        default_index = self.milling_voltage_combo.findData(30e3)
        if default_index >= 0:
            self.milling_voltage_combo.setCurrentIndex(default_index)
        layout.addWidget(self.milling_voltage_combo, row, 1)
        row += 1

        # Milling Channel
        layout.addWidget(QLabel("Milling Channel"), row, 0)
        self.milling_channel_combo = QComboBox()
        self.milling_channel_combo.addItem("Ion Beam", BeamType.ION)
        self.milling_channel_combo.addItem("Electron Beam", BeamType.ELECTRON)
        layout.addWidget(self.milling_channel_combo, row, 1)
        row += 1

        # Patterning Mode
        layout.addWidget(QLabel("Patterning Mode"), row, 0)
        self.patterning_mode_combo = QComboBox()
        for mode in PATTERNING_MODES:
            self.patterning_mode_combo.addItem(mode)
        self.patterning_mode_combo.setCurrentText("Serial")
        layout.addWidget(self.patterning_mode_combo, row, 1)
        row += 1

        # Advanced settings (initially hidden)
        
        # Spot Size
        self.spot_size_label = QLabel("Spot Size")
        layout.addWidget(self.spot_size_label, row, 0)
        self.spot_size_spinbox = QDoubleSpinBox()
        spot_config = WIDGET_CONFIG["spot_size"]
        self.spot_size_spinbox.setRange(*spot_config["range"])
        self.spot_size_spinbox.setDecimals(spot_config["decimals"])
        self.spot_size_spinbox.setSingleStep(spot_config["step"])
        self.spot_size_spinbox.setValue(spot_config["default"])
        self.spot_size_spinbox.setSuffix(spot_config["suffix"])
        layout.addWidget(self.spot_size_spinbox, row, 1)
        row += 1

        # Rate
        self.rate_label = QLabel("Milling Rate")
        layout.addWidget(self.rate_label, row, 0)
        self.rate_spinbox = QDoubleSpinBox()
        rate_config = WIDGET_CONFIG["rate"]
        self.rate_spinbox.setRange(*rate_config["range"])
        self.rate_spinbox.setDecimals(rate_config["decimals"])
        self.rate_spinbox.setSingleStep(rate_config["step"])
        self.rate_spinbox.setValue(rate_config["default"])
        self.rate_spinbox.setSuffix(rate_config["suffix"])
        layout.addWidget(self.rate_spinbox, row, 1)
        row += 1

        # Spacing
        self.spacing_label = QLabel("Spacing")
        layout.addWidget(self.spacing_label, row, 0)
        self.spacing_spinbox = QDoubleSpinBox()
        spacing_config = WIDGET_CONFIG["spacing"]
        self.spacing_spinbox.setRange(*spacing_config["range"])
        self.spacing_spinbox.setDecimals(spacing_config["decimals"])
        self.spacing_spinbox.setSingleStep(spacing_config["step"])
        self.spacing_spinbox.setValue(spacing_config["default"])
        self.spacing_spinbox.setSuffix(spacing_config["suffix"])
        layout.addWidget(self.spacing_spinbox, row, 1)
        row += 1

        # Application File
        self.application_file_label = QLabel("Application File")
        layout.addWidget(self.application_file_label, row, 0)
        self.application_file_combo = QComboBox()
        for app_file in APPLICATION_FILES:
            self.application_file_combo.addItem(app_file)
        self.application_file_combo.setCurrentText("Si")
        layout.addWidget(self.application_file_combo, row, 1)
        row += 1

        # Preset
        self.preset_label = QLabel("Preset")
        layout.addWidget(self.preset_label, row, 0)
        self.preset_combo = QComboBox()
        for preset in PRESETS:
            self.preset_combo.addItem(preset)
        self.preset_combo.setCurrentText("30 keV; 2nA")
        layout.addWidget(self.preset_combo, row, 1)
        row += 1

        # Boolean options
        self.acquire_images_check = QCheckBox("Acquire Images")
        layout.addWidget(self.acquire_images_check, row, 0, 1, 2)
        row += 1

    def _connect_signals(self):
        """Connect widget signals to their respective handlers."""
        self.milling_current_combo.currentIndexChanged.connect(self._emit_settings_changed)
        self.dwell_time_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.hfw_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.milling_voltage_combo.currentIndexChanged.connect(self._emit_settings_changed)
        self.milling_channel_combo.currentIndexChanged.connect(self._emit_settings_changed)
        self.patterning_mode_combo.currentTextChanged.connect(self._emit_settings_changed)
        self.spot_size_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.rate_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.spacing_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.application_file_combo.currentTextChanged.connect(self._emit_settings_changed)
        self.preset_combo.currentTextChanged.connect(self._emit_settings_changed)
        self.acquire_images_check.toggled.connect(self._emit_settings_changed)

    def _update_advanced_visibility(self):
        """Show/hide advanced settings based on the show_advanced flag.

        Advanced settings include: spot size, rate, spacing, application file, preset.
        """
        self.spot_size_label.setVisible(self._show_advanced)
        self.spot_size_spinbox.setVisible(self._show_advanced)
        self.rate_label.setVisible(self._show_advanced)
        self.rate_spinbox.setVisible(self._show_advanced)
        self.spacing_label.setVisible(self._show_advanced)
        self.spacing_spinbox.setVisible(self._show_advanced)
        self.application_file_label.setVisible(self._show_advanced)
        self.application_file_combo.setVisible(self._show_advanced)
        self.preset_label.setVisible(self._show_advanced)
        self.preset_combo.setVisible(self._show_advanced)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self._show_advanced = show_advanced
        self._update_advanced_visibility()

    def toggle_advanced(self):
        """Toggle the visibility of advanced settings.

        Switches between showing and hiding the advanced controls.
        """
        self.set_show_advanced(not self._show_advanced)

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self._show_advanced

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> FibsemMillingSettings:
        """Get the current FibsemMillingSettings from the widget values.

        Returns:
            FibsemMillingSettings object with values from the UI controls.
            Units are converted from display units to SI units.
            Updates only the fields controlled by this widget, preserving
            all other fields from the stored settings.
        """
        # Update only the fields controlled by this widget
        self._settings.milling_current = self.milling_current_combo.currentData()
        self._settings.dwell_time = self.dwell_time_spinbox.value() * MICRO_TO_SI  # Convert μs to s
        self._settings.hfw = self.hfw_spinbox.value() * MICRO_TO_SI  # Convert μm to m
        self._settings.milling_voltage = self.milling_voltage_combo.currentData()
        self._settings.milling_channel = self.milling_channel_combo.currentData()
        self._settings.patterning_mode = self.patterning_mode_combo.currentText()
        self._settings.spot_size = self.spot_size_spinbox.value() * 1e-9  # Convert nm to m
        self._settings.rate = self.rate_spinbox.value() * 1e-21  # Convert mm³/nA/s to m³/A/s
        self._settings.spacing = self.spacing_spinbox.value()
        self._settings.application_file = self.application_file_combo.currentText()
        self._settings.preset = self.preset_combo.currentText()
        self._settings.acquire_images = self.acquire_images_check.isChecked()
        
        return self._settings

    def update_from_settings(self, settings: FibsemMillingSettings):
        """Update all widget values from a FibsemMillingSettings object.

        Args:
            settings: FibsemMillingSettings object to load values from.
                     Units are converted from SI units to display units.
        """
        self._settings = settings

        # Block signals to prevent recursive updates
        self.blockSignals(True)

        # Set milling current
        current_index = self.milling_current_combo.findData(settings.milling_current)
        if current_index >= 0:
            self.milling_current_combo.setCurrentIndex(current_index)

        self.dwell_time_spinbox.setValue(settings.dwell_time * SI_TO_MICRO)  # Convert s to μs
        self.hfw_spinbox.setValue(settings.hfw * SI_TO_MICRO)  # Convert m to μm

        # Set milling voltage
        voltage_index = self.milling_voltage_combo.findData(settings.milling_voltage)
        if voltage_index >= 0:
            self.milling_voltage_combo.setCurrentIndex(voltage_index)
        
        # Set milling channel
        channel_index = self.milling_channel_combo.findData(settings.milling_channel)
        if channel_index >= 0:
            self.milling_channel_combo.setCurrentIndex(channel_index)

        self.patterning_mode_combo.setCurrentText(settings.patterning_mode)
        self.spot_size_spinbox.setValue(settings.spot_size * 1e9)  # Convert m to nm
        self.rate_spinbox.setValue(settings.rate * 1e21)  # Convert m³/A/s to mm³/nA/s
        self.spacing_spinbox.setValue(settings.spacing)
        self.application_file_combo.setCurrentText(settings.application_file)
        self.preset_combo.setCurrentText(settings.preset)
        self.acquire_images_check.setChecked(settings.acquire_images)

        # Update visibility based on settings
        self._update_advanced_visibility()

        self.blockSignals(False)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication, QCheckBox, QPushButton, QVBoxLayout

    app = QApplication(sys.argv)

    # Create main window
    main_widget = QWidget()
    layout = QVBoxLayout()
    main_widget.setLayout(layout)

    # Create the MillingSettings widget
    settings_widget = MillingSettingsWidget(show_advanced=False)
    layout.addWidget(settings_widget)

    # Add advanced settings toggle checkbox
    advanced_checkbox = QCheckBox("Show Advanced Settings")
    advanced_checkbox.setChecked(settings_widget.get_show_advanced())
    advanced_checkbox.toggled.connect(settings_widget.set_show_advanced)
    layout.addWidget(advanced_checkbox)

    # Add a button to print current settings
    def print_settings():
        settings = settings_widget.get_settings()
        print("Current MillingSettings:")
        for field, value in settings.__dict__.items():
            print(f"  {field}: {value}")

    print_button = QPushButton("Print Current Settings")
    print_button.clicked.connect(print_settings)
    layout.addWidget(print_button)

    # Connect to settings change signal
    def on_settings_changed(settings: FibsemMillingSettings):
        print(f"Settings changed - {settings}")

    settings_widget.settings_changed.connect(on_settings_changed)

    main_widget.setWindowTitle("MillingSettings Widget Test")
    main_widget.show()

    sys.exit(app.exec_())