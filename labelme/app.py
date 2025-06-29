# -*- coding: utf-8 -*-

import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser

import imgviz
import natsort
import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


from labelme import __appname__
from labelme._automation import bbox_from_text
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.shape import Shape
from labelme.widgets import AiPromptWidget
from labelme.widgets import AiLabelWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme._automation.bbox_from_text import inference

from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()
          

class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])  # type: ignore[assignment]
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])  # type: ignore[assignment]
        Shape.select_line_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}")
        )

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),  # type: ignore[attr-defined]
            Qt.Horizontal: scrollArea.horizontalScrollBar(),  # type: ignore[attr-defined]
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)  # type: ignore[attr-defined]
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)  # type: ignore[attr-defined]
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)  # type: ignore[attr-defined]
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)  # type: ignore[attr-defined]


        # Actions
        action = functools.partial(utils.newAction, self)
        
        # ——— AI-Label button ———
        #aiLabelAction = action(
        #    self.tr("AI &Label"),              # text (with mnemonic)
        #    self.submit_ai_label,              # slot
        #    None,                              # shortcut (or e.g. "Ctrl+Shift+L")
        #    "robot",                           # icon name (choose one you like)
        #    self.tr("Automatically label using VLM")
        #)
        '''
        till here
        '''
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),  # type: ignore[attr-defined]
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                model_name=self._selectAiModelComboBox.itemData(  # type: ignore[has-type]
                    self._selectAiModelComboBox.currentIndex()  # type: ignore[has-type]
                )
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "objects",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                model_name=self._selectAiModelComboBox.itemData(  # type: ignore[has-type]
                    self._selectAiModelComboBox.currentIndex()  # type: ignore[has-type]
                )
            )
            if self.canvas.createMode == "ai_mask"
            else None
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)  # type: ignore[union-attr]
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)  # type: ignore[attr-defined]
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        # ——— Describe action for context menu ————————————————————
        self.describeAction = action(
            self.tr("Describe"),
            slot=self.on_describe_shape,
            shortcut=None,
            icon="info",
            tip=self.tr("Describe contents of the selected bounding box"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(  # type: ignore[assignment,method-assign]
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
                None,
                self.describeAction,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)  # type: ignore[attr-defined]

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,  # type: ignore[attr-defined]
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,  # type: ignore[attr-defined]
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))  # type: ignore[attr-defined]
        utils.addActions(
            self.menus.view,  # type: ignore[attr-defined]
            (
                #self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)  # type: ignore[attr-defined]

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)  # type: ignore[attr-defined]
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())  # type: ignore[union-attr]
        #
        selectAiModelLabel = QtWidgets.QLabel(self.tr("SAM Mask Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)  # type: ignore[attr-defined]
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)  # type: ignore[union-attr]
        #
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)  # type: ignore[union-attr]
        MODEL_NAMES: list[tuple[str, str]] = [
            ("efficientsam:10m", "EfficientSam (speed)"),
            ("efficientsam:latest", "EfficientSam (accuracy)"),
            ("sam:100m", "SegmentAnything (speed)"),
            ("sam:300m", "SegmentAnything (balanced)"),
            ("sam:latest", "SegmentAnything (accuracy)"),
            ("sam2:small", "Sam2 (speed)"),
            ("sam2:latest", "Sam2 (balanced)"),
            ("sam2:large", "Sam2 (accuracy)"),
        ]
        for model_name, model_ui_name in MODEL_NAMES:
            self._selectAiModelComboBox.addItem(model_ui_name, userData=model_name)
        model_ui_names: list[str] = [model_ui_name for _, model_ui_name in MODEL_NAMES]
        if self._config["ai"]["default"] in model_ui_names:
            model_index = model_ui_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda index: self.canvas.initializeAiModel(
                model_name=self._selectAiModelComboBox.itemData(index)
            )
            if self.canvas.createMode in ["ai_polygon", "ai_mask"]
            else None
        )



        """ Add ai label """
        # ─── AI Label Prompt (embedded in toolbar) ────────────────────────────────
        self._ai_label_prompt_widget: QtWidgets.QWidget = AiLabelWidget(
            on_submit=self.submit_ai_label, parent=self
        )
        ai_label_prompt_action = QtWidgets.QWidgetAction(self)
        ai_label_prompt_action.setDefaultWidget(self._ai_label_prompt_widget)
        # ───────────────────────────────────────────────────────────────────────────




        self._ai_prompt_widget: QtWidgets.QWidget = AiPromptWidget(
            on_submit=self.submit_custom_ai_prompt, parent=self
        )
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)
        
        self.tools = self.toolbar("Tools")
        self.actions.tool = (  # type: ignore[attr-defined]
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            save,
            deleteFile,
            None,
            createMode,
            editMode,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            fitWindow,
            zoom,
            None,
            selectAiModel,
            None,
            #ai_prompt_action,
            ai_label_prompt_action,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)  # type: ignore[union-attr]
        self.statusBar().show()  # type: ignore[union-attr]

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []  # type: ignore[var-annotated]
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},  # type: ignore[attr-defined]
            Qt.Vertical: {},  # type: ignore[attr-defined]
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()
        
        '''
        # ─── Custom AI Prompt dock with inline Submit button ──────────────────────
        self.prompt_dock = QtWidgets.QDockWidget(self.tr("VLM Prompt"), self)
        self.prompt_dock.setObjectName("PromptDock")

        # Container widget for prompt editor + submit button
        container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(container)
        vlayout.setContentsMargins(4, 4, 4, 4)

        # 1) Multi-line prompt editor
        self.promptEditor = QtWidgets.QPlainTextEdit()
        self.promptEditor.setPlaceholderText(self.tr("Enter AI prompt here..."))
        # Mark dirty when user types
        self.promptEditor.textChanged.connect(lambda: self.setDirty())
        vlayout.addWidget(self.promptEditor)

        # 2) Submit button
        submit_btn = QtWidgets.QPushButton(self.tr("Submit"))
        submit_btn.clicked.connect(self.submit_custom_ai_prompt)
        vlayout.addWidget(submit_btn)

        self.prompt_dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prompt_dock)
        '''
        # ─── Custom AI Prompts dock (history + Add/Remove) ──────────────────────
        from PyQt5.QtWidgets import QListWidget, QInputDialog

        self.prompt_dock = QtWidgets.QDockWidget(self.tr("VLM Prompts"), self)
        self.prompt_dock.setObjectName("PromptHistoryDock")
        container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(container)
        vlayout.setContentsMargins(4, 4, 4, 4)

        # 1) History list
        self.promptListWidget = QListWidget()
        vlayout.addWidget(self.promptListWidget)

        # 2) Add/Remove buttons
        hlayout = QtWidgets.QHBoxLayout()
        self.addPromptBtn = QtWidgets.QPushButton(self.tr("Add Prompt"))
        self.removePromptBtn = QtWidgets.QPushButton(self.tr("Remove Prompt"))
        hlayout.addWidget(self.addPromptBtn)
        hlayout.addWidget(self.removePromptBtn)
        vlayout.addLayout(hlayout)

        self.prompt_dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prompt_dock)

        # Internal store of prompt/description pairs
        self.prompt_history: List[Dict[str,str]] = []

        # Wire up UI
        self.addPromptBtn.clicked.connect(self.on_add_prompt)
        self.removePromptBtn.clicked.connect(self.on_remove_prompt)
        self.promptListWidget.currentRowChanged.connect(self.on_prompt_selected)
        self.promptListWidget.itemDoubleClicked.connect(self.on_edit_prompt)
        # ---- Ensure description_dock and descriptionEditor are initialized and added to the layout
        self.description_dock = QtWidgets.QDockWidget(self.tr("VLM Description"), self)
        self.description_dock.setObjectName("DescriptionDock")
        self.description_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )

        self.descriptionEditor = QtWidgets.QPlainTextEdit()
        self.descriptionEditor.setPlaceholderText(self.tr("VLM description output and editing area..."))
        self.descriptionEditor.setReadOnly(False)  # Make it editable
        self.description_dock.setWidget(self.descriptionEditor)

        self.addDockWidget(Qt.RightDockWidgetArea, self.description_dock)
        self.descriptionEditor.textChanged.connect(lambda: self.setDirty())
        # Place it, for example, below the Custom AI Prompt dock
        #self.splitDockWidget(self._ai_prompt_widget, self._ai_label_dock, Qt.Vertical)

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)  # type: ignore[union-attr]
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # type: ignore[attr-defined]
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)  # type: ignore[attr-defined]
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu  # type: ignore[attr-defined]
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()  # type: ignore[attr-defined]
        actions = (
            self.actions.createMode,  # type: ignore[attr-defined]
            self.actions.createRectangleMode,  # type: ignore[attr-defined]
            self.actions.createCircleMode,  # type: ignore[attr-defined]
            self.actions.createLineMode,  # type: ignore[attr-defined]
            self.actions.createPointMode,  # type: ignore[attr-defined]
            self.actions.createLineStripMode,  # type: ignore[attr-defined]
            self.actions.createAiPolygonMode,  # type: ignore[attr-defined]
            self.actions.createAiMaskMode,  # type: ignore[attr-defined]
            self.actions.editMode,  # type: ignore[attr-defined]
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)  # type: ignore[attr-defined]

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)  # type: ignore[attr-defined]

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():  # type: ignore[attr-defined]
            label_file = osp.splitext(self.imagePath)[0] + ".json"  # type: ignore[arg-type]
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)  # type: ignore[attr-defined]
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)  # type: ignore[attr-defined]
        self.actions.createMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createRectangleMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createCircleMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createLineMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createPointMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createLineStripMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createAiPolygonMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createAiMaskMode.setEnabled(True)  # type: ignore[attr-defined]
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)  # type: ignore[attr-defined]
        else:
            self.actions.deleteFile.setEnabled(False)  # type: ignore[attr-defined]

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:  # type: ignore[attr-defined]
            z.setEnabled(value)
        for action in self.actions.onLoadActive:  # type: ignore[attr-defined]
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)  # type: ignore[union-attr]

    def submit_custom_ai_prompt(self):
        from labelme._automation.bbox_from_text import inference
        prompt_text = self.promptEditor.toPlainText().strip()

        if not prompt_text:
            return self.errorMessage("Error", "Please enter a prompt in the dock.")
        
        self.status("Running AI description…")
        try:
            description_text = inference(self.imagePath, prompt_text)
            # 4. Populate the description dock
            self.descriptionEditor.setPlainText(description_text)
            self.setDirty()
            self.status("AI description complete.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.errorMessage("Error", f"AI description failed:\n{e}")
    '''
    def submit_custom_ai_prompt(self):
        from labelme._automation.bbox_from_text import inference
        
        # 1. Grab the user’s prompt text
        prompt_text = self._ai_prompt_widget.get_text_prompt().strip()
        if hasattr(self, "promptEditor"):
            self.promptEditor.setPlainText(prompt_text)
        if not prompt_text:
            return self.errorMessage("Error", "Please enter a prompt.")

        # 2. Make sure we have an image on-disk
        if not hasattr(self, 'imagePath') or not self.imagePath:
            return self.errorMessage("Error", "Open an image first.")

        self.status("Running AI description…")
        try:
            # 3. Call inference() with the file path
            description_text = inference(self.imagePath, prompt_text)

            # 4. Populate the description dock
            self.descriptionEditor.setPlainText(description_text)

            # 5. Mark document dirty so “Save” writes it out
            self.setDirty()
            self.status("AI description complete.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.errorMessage("Error", f"AI description failed:\n{e}")
    '''
    '''
    def _submit_ai_prompt(self, _) -> None:
        texts = self._ai_prompt_widget.get_text_prompt().split(",")
        boxes, scores, labels = bbox_from_text.get_bboxes_from_texts(
            model="yoloworld",
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            texts=texts,
        )

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle" or shape.label not in texts:
                continue
            box = np.array(
                [
                    shape.points[0].x(),
                    shape.points[0].y(),
                    shape.points[1].x(),
                    shape.points[1].y(),
                ],
                dtype=np.float32,
            )
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [1.01]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels = bbox_from_text.nms_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
            score_threshold=self._ai_prompt_widget.get_score_threshold(),
            max_num_detections=100,
        )

        keep = scores != 1.01
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        shape_dicts: list[dict] = bbox_from_text.get_shapes_from_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
        )

        shapes: list[Shape] = []
        for shape_dict in shape_dicts:
            shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                description=shape_dict["description"],
            )
            for point in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(*point))
            shapes.append(shape)

        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setDirty()

    '''

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)  # type: ignore[attr-defined]

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)  # type: ignore[attr-defined]
        self.actions.undoLastPoint.setEnabled(drawing)  # type: ignore[attr-defined]
        self.actions.undo.setEnabled(not drawing)  # type: ignore[attr-defined]
        self.actions.delete.setEnabled(not drawing)  # type: ignore[attr-defined]

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,  # type: ignore[attr-defined]
            "rectangle": self.actions.createRectangleMode,  # type: ignore[attr-defined]
            "circle": self.actions.createCircleMode,  # type: ignore[attr-defined]
            "point": self.actions.createPointMode,  # type: ignore[attr-defined]
            "line": self.actions.createLineMode,  # type: ignore[attr-defined]
            "linestrip": self.actions.createLineStripMode,  # type: ignore[attr-defined]
            "ai_polygon": self.actions.createAiPolygonMode,  # type: ignore[attr-defined]
            "ai_mask": self.actions.createAiMaskMode,  # type: ignore[attr-defined]
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)  # type: ignore[attr-defined]

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles  # type: ignore[attr-defined]
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))  # type: ignore[attr-defined]

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)  # type: ignore[attr-defined,union-attr]
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(True)  # type: ignore[union-attr]
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(False)  # type: ignore[union-attr]
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()  # type: ignore[no-redef]

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                item.setText(
                    '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                        html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                    )
                )
            else:
                item.setText("{} ({})".format(shape.label, shape.group_id))
            self.setDirty()
            if self.uniqLabelList.findItemByLabel(shape.label) is None:
                item = self.uniqLabelList.createItemFromLabel(shape.label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(shape.label)
                self.uniqLabelList.setItemLabel(item, shape.label, rgb)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.duplicate.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.copy.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.edit.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.describeAction.setEnabled(n_selected == 1)


    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags: dict = shape["flags"] or {}
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()  # type: ignore[union-attr]
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # type: ignore[attr-defined]
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)  # type: ignore[attr-defined]
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            key = item.text()  # type: ignore[union-attr]
            flag = item.checkState() == Qt.Checked  # type: ignore[attr-defined,union-attr]
            flags[key] = flag

        other_data = self.otherData or {}
        other_data["vlm_description"] = self.descriptionEditor.toPlainText()
        #other_data["custom_ai_prompt"] = self.promptEditor.toPlainText()
        other_data["prompt_history"] = self.prompt_history
        self.otherData = other_data
        
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))  # type: ignore[arg-type]
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                otherData=other_data,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)  # type: ignore[attr-defined]
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)  # type: ignore[attr-defined]
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)  # type: ignore[attr-defined]

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)  # type: ignore[attr-defined]

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)  # type: ignore[attr-defined]
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)  # type: ignore[attr-defined]
            self.actions.undoLastPoint.setEnabled(False)  # type: ignore[attr-defined]
            self.actions.undo.setEnabled(True)  # type: ignore[attr-defined]
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units  # type: ignore[union-attr]
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))  # type: ignore[union-attr]
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)  # type: ignore[attr-defined]
        self.actions.fitWindow.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,  # type: ignore[attr-defined]
                self.scrollBars[Qt.Horizontal].value() + x_shift,  # type: ignore[attr-defined,union-attr]
            )
            self.setScroll(
                Qt.Vertical,  # type: ignore[attr-defined]
                self.scrollBars[Qt.Vertical].value() + y_shift,  # type: ignore[attr-defined,union-attr]
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)  # type: ignore[attr-defined]

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked  # type: ignore[attr-defined]
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)  # type: ignore[attr-defined]
    '''
    Modified loadfile
    '''
    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,  # type: ignore[arg-type]
            )
            self.otherData = self.labelFile.otherData or {}

            # === New code to load VLM description into descriptionEditor ===
            description_text = self.otherData.get("vlm_description", "")
            if hasattr(self, "descriptionEditor"):
                self.descriptionEditor.setPlainText(description_text)

            # === New code to load AI prompt into promptEditor ===
            self.prompt_history = self.otherData.get("prompt_history", [])
            self.promptListWidget.clear()
            for entry in self.prompt_history:
                self.promptListWidget.addItem(entry["prompt"])
            # select first by default
            if self.prompt_history:
                self.promptListWidget.setCurrentRow(0)
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
            # might cause problem
            self.prompt_history.clear()
            self.promptListWidget.clear()
            # Clear the description editor if no label file
            if hasattr(self, "descriptionEditor"):
                self.descriptionEditor.clear()

            #if hasattr(self, "promptEditor"):
            #    self.promptEditor.clear()
            #self.promptEditor.clear()
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        return True

    '''
    till here
    '''
    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e  # type: ignore[union-attr]
        h1 = self.centralWidget().height() - e  # type: ignore[union-attr]
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0  # type: ignore[union-attr]
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)  # type: ignore[attr-defined]

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier  # type: ignore[attr-defined]
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier  # type: ignore[attr-defined]
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(  # type: ignore[union-attr]
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()  # type: ignore[union-attr]

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename  # type: ignore[assignment]
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)  # type: ignore[attr-defined]

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, " "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)  # type: ignore[attr-defined,union-attr]

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:  # type: ignore[union-attr]
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, " "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())  # type: ignore[union-attr]
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)  # type: ignore[attr-defined]
            else:
                item.setCheckState(Qt.Unchecked)  # type: ignore[attr-defined]
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)  # type: ignore[attr-defined]
            self.actions.openPrevImg.setEnabled(True)  # type: ignore[attr-defined]

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.openPrevImg.setEnabled(True)  # type: ignore[attr-defined]

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)  # type: ignore[attr-defined]
            else:
                item.setCheckState(Qt.Unchecked)  # type: ignore[attr-defined]
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
    """
    New thing, submit ai label
    """
    """
    def submit_ai_label(self):
        from labelme._automation.bbox_from_text import get_vlm_shapes

        if self.image.isNull():
            return self.errorMessage("Error", "Open an image first.")
        prompt = "Outline the position of each and output all the coordinates in JSON format."
        self.status("Running AI Label…")
        try:
            image_np = utils.img_qt_to_arr(self.image)[:, :, :3]
            shape_dicts, description_text = get_vlm_shapes(image_np, prompt)

            shapes = []
            for d in shape_dicts:
                shape = Shape(
                    label=d["label"],
                    shape_type=d["shape_type"],
                    description=d.get("description", "")
                )
                for x,y in d["points"]:
                    shape.addPoint(QtCore.QPointF(x, y))
                shapes.append(shape)

            self.canvas.storeShapes()
            self.loadShapes(shapes, replace=False)
            # push any text return into the description editor for review
            self.descriptionEditor.setPlainText(description_text)
            self.setDirty()
            self.status("AI Label complete.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.errorMessage("Error", f"AI auto‐labeling failed:\n{e}")
    """
    def submit_ai_label(self):
        from labelme._automation.bbox_from_text import get_vlm_shapes

        # 1. Read whatever the user typed in the “AI Label Prompt” dock:
        prompt_text = self._ai_label_prompt_widget.get_text_prompt().strip()
        if not prompt_text:
            return self.errorMessage("Error", "Please enter a prompt for AI Label.")

        # 2. Append “output in json format” to force JSON bbox output:
        prompt = f" outline the position of each {prompt_text.strip()} and output all the coordinates in JSON format as such: \"bbox\": [..., ..., ..., ...], \"label\": \"the label\""

        # 3. Make sure an image is open:
        if self.image.isNull():
            return self.errorMessage("Error", "Open an image first.")

        self.status("Running AI Label…")
        try:
            # Convert the currently displayed QImage → NumPy array:
            image_np = utils.img_qt_to_arr(self.image)[:, :, :3]

            # Call your VLM routine:
            shape_dicts, description_text = get_vlm_shapes(image_np, prompt)

            # Build LabelMe shapes from the returned bbox dicts:
            shapes = []
            for d in shape_dicts:
                shape = Shape(
                    label=d["label"],
                    shape_type=d["shape_type"],
                    description=d.get("description", "")
                )
                for x, y in d["points"]:
                    shape.addPoint(QtCore.QPointF(x, y))
                shapes.append(shape)

            # Push them onto the canvas:
            self.canvas.storeShapes()
            self.loadShapes(shapes, replace=False)

            # Any “description” returned from the model goes into the description dock:
            if description_text and description_text.strip():
                old = self.descriptionEditor.toPlainText().strip()
                new = description_text.strip()
                if old:
                    self.descriptionEditor.setPlainText(old + "\n\n" + new)
                else:
                    self.descriptionEditor.setPlainText(new)
            self.setDirty()
            self.status("AI Label complete.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.errorMessage("Error", f"AI auto‐labeling failed:\n{e}")
    
    def on_add_prompt(self):
        # Ask the user for a new prompt
        from PyQt5.QtWidgets import QInputDialog
        prompt, ok = self._get_prompt_from_dialog(self.tr("New AI Prompt"), "")
        if not ok or not prompt.strip():
            return

        # Run the model
        self.status(self.tr("Running AI description…"))
        try:
            desc = inference(self.imagePath, prompt)
        except Exception as e:
            self.errorMessage(self.tr("Error"), str(e))
            return

        # Store and display
        entry = {"prompt": prompt, "description": desc}
        self.prompt_history.append(entry)
        self.promptListWidget.addItem(prompt)
        row = len(self.prompt_history) - 1
        self.promptListWidget.setCurrentRow(row)

        # Show in the VLM description dock
        self.descriptionEditor.setPlainText(desc)
        self.setDirty()

    def on_remove_prompt(self):
        row = self.promptListWidget.currentRow()
        if row < 0:
            return
        # Remove from UI + data
        self.prompt_history.pop(row)
        self.promptListWidget.takeItem(row)
        self.setDirty()

        # Select previous or clear
        if self.prompt_history:
            new_row = min(row, len(self.prompt_history)-1)
            self.promptListWidget.setCurrentRow(new_row)
        else:
            self.descriptionEditor.clear()

    def on_prompt_selected(self, row: int):
        if row < 0 or row >= len(self.prompt_history):
            return
        # Display the associated description
        entry = self.prompt_history[row]
        # 1) Show the saved description
        desc = entry["description"]
        self.descriptionEditor.setPlainText(desc)

        # 2) If this entry came from a Describe call, highlight that box
        bbox = entry.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            # find the matching shape on the canvas
            match = None
            for shape in self.canvas.shapes:
                if shape.shape_type == "rectangle":
                    p1, p2 = shape.points
                    if (
                        int(p1.x()) == x1 and int(p1.y()) == y1
                        and int(p2.x()) == x2 and int(p2.y()) == y2
                    ):
                        match = shape
                        break
            # now select it (and clear any other selection)
            if match:
                self.canvas.selectedShapes = [match]
                self.shapeSelectionChanged([match])
        else:
            # no bbox → just clear selection
            self.canvas.selectedShapes = []
            self.shapeSelectionChanged([])

    def on_edit_prompt(self, item: QtWidgets.QListWidgetItem):
        """Double-click handler: edit an existing prompt, re-run inference."""
        row = self.promptListWidget.row(item)
        old = self.prompt_history[row]["prompt"]
        # Open the same dialog, seeded with the old prompt
        new_prompt, ok = self._get_prompt_from_dialog(self.tr("Edit AI Prompt"), old)
        if not ok or not new_prompt.strip() or new_prompt == old:
            return

        # Re-run the model on the new prompt
        try:
            desc = inference(self.imagePath, new_prompt)
        except Exception as e:
            self.errorMessage(self.tr("Error"), f"AI inference failed:\n{e}")
            return

        # Update history, UI list, and description dock
        self.prompt_history[row] = {"prompt": new_prompt, "description": desc}
        item.setText(new_prompt)
        self.promptListWidget.setCurrentRow(row)
        self.descriptionEditor.setPlainText(desc)
        self.setDirty()

    def _get_prompt_from_dialog(self, title: str, initial: str):
        """
        Helper: open a QDialog with a QPlainTextEdit for multi-line prompts.
        Returns (text, accepted_bool).
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dlg)
        edit = QtWidgets.QPlainTextEdit()
        edit.setPlainText(initial)
        edit.setMinimumSize(400, 150)
        layout.addWidget(edit)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        res = dlg.exec_() == QtWidgets.QDialog.Accepted
        return edit.toPlainText().strip(), res
    
    def on_describe_shape(self):
        """
        Right-click → Describe: highlight the selected rectangle in red,
        ask the VLM 'What is in the red bounding box?', then record
        that prompt+answer in your prompt history.
        """
        from labelme._automation.bbox_from_text import inference
        import tempfile, os
        # 1) Make sure exactly one rectangle is selected
        shapes = self.canvas.selectedShapes
        if len(shapes) != 1 or shapes[0].shape_type != "rectangle":
            return

        # 2) Extract box coordinates
        shape = shapes[0]
        p1, p2 = shape.points
        x1, y1 = int(p1.x()), int(p1.y())
        x2, y2 = int(p2.x()), int(p2.y())

        # 3) Copy the image and draw a thick red rectangle
        img_copy = QtGui.QImage(self.image)  # deep copy
        painter = QtGui.QPainter(img_copy)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.end()

        #img_copy.save(os.path.join(os.path.expanduser("~"), "Desktop", "bbox_debug.png"),"PNG")


        # 4) Convert to NumPy and run inference
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        img_copy.save(tmp_path, "PNG")
        prompt = "What is in the bounding box?"
        try:
            desc = inference(tmp_path, prompt)
        except Exception as e:
            return self.errorMessage(self.tr("Error"), str(e))
        finally:
            try: os.remove(tmp_path)
            except OSError: pass

        # 5) Append to your prompt history UI
        entry = {"prompt": prompt, "description": desc, "bbox": [x1, y1, x2, y2]}
        self.prompt_history.append(entry)
        self.promptListWidget.addItem(prompt)
        idx = len(self.prompt_history) - 1
        self.promptListWidget.setCurrentRow(idx)

        # 6) Show in the description dock and mark dirty
        self.descriptionEditor.setPlainText(desc)
        self.setDirty()
        self.status(self.tr("Describe complete."))
