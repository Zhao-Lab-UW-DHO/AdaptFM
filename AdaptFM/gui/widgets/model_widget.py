from magicgui import magicgui
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog,
    QPushButton, QFormLayout,QScrollArea
)
from pathlib import Path
from AdaptFM.model.registry import MODEL_REGISTRY
from AdaptFM.model.fmSpec import FoundationModelSpec


class ModelWorkflowWidget:
    TAG_LABEL = "Tag"   # overridden by subclasses

    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
        self.model = None
        self.dataset_dir = None
        self.param_widgets = {}

        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)

        self._build_base()

    # ---------- UI BUILD ----------

    def _build_base(self):
        self.params_form = QFormLayout()

        params_widget = QWidget()
        params_widget.setLayout(self.params_form)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(params_widget)
        scroll.setMaximumHeight(400)  # or ~50–60% of window height


        self._build_model_selector()
        self._build_dataset_selector()
        self.layout.addWidget(scroll)

        self._build_run_button()

    def _build_model_selector(self):
        @magicgui(model={"choices": list(MODEL_REGISTRY.keys())},
                  call_button=False)
        def select_model(model: str):
            self._on_model_selected(model)

        self.select_model_widget = select_model
        self.layout.addWidget(select_model.native)

        select_model.model.changed.connect(self._on_model_selected)
        default_name = select_model.model.value
        if default_name is not None:
            self._on_model_selected(default_name)


    def _build_tag_input(self):
        if hasattr(self, "tag_widget"):
            return
        
        if hasattr(self,'skip_tag'):
            return

        @magicgui(call_button="Load parameters",
                tag={"label": self.TAG_LABEL})
        def load_tag(tag: str):
            self._load_tunable_params(tag)

        self.tag_widget = load_tag
        self.tag_widget.visible = True
        self.tag_widget.show()


    def _build_dataset_selector(self):
        btn = QPushButton("Select dataset folder")
        btn.clicked.connect(self._select_dataset_folder)
        self.layout.addWidget(btn)

    def _build_run_button(self):
        btn = QPushButton("Run")
        btn.clicked.connect(self._run)
        self.layout.addWidget(btn)

    # ---------- LOGIC ----------

    def _on_model_selected(self, model_name):
        self.model = MODEL_REGISTRY[model_name]
        self._clear_params()
        self._on_model_changed()

        if isinstance(self.model, FoundationModelSpec):
            # needs a tag first
            self._build_tag_input()
        else:
            # nnUNetv2, classical models, etc
            self._load_tunable_params(tag=None)

    def _load_tunable_params(self, tag):

        self._clear_params()

        schema = self.model.tunable_params(tag=tag)
        
        for name, spec in schema.items():
            w = self._make_param_widget(name, spec)
            self.param_widgets[name] = w
            self.params_form.addRow(name,w.native)
    

    def _make_param_widget(self, name, spec):
        from magicgui.widgets import create_widget

        # work on a copy so you don't mutate the original spec
        spec = dict(spec)

        # pull out the semantic keys you use to define the widget
        annotation = spec.pop("type", None)
        value = spec.pop("default", None)

        # everything left in spec is safe to pass as options
        return create_widget(
            value=value,
            annotation=annotation,
            options=spec,
        )


    def _select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(None, "Select dataset folder")
        if folder:
            self.dataset_manager.load_from_folder(folder)
            self.dataset_dir = Path(folder)

    def collect_params(self):
        return {k: w.value for k, w in self.param_widgets.items()}

    def _clear_params(self):
        while self.params_form.count():
            self.params_form.removeRow(0)
        self.param_widgets.clear()

    # ---------- OVERRIDES ----------

    def _run(self):
        """Implemented by subclasses"""
        raise NotImplementedError
    
    def _on_model_changed(self):
        """Hook for subclasses to reset state when the model changes."""
        pass
