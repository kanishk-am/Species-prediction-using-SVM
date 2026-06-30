"""
Iris Species Predictor — SVM (Support Vector Machine)
========================================================
Enter sepal & petal measurements → get predicted species + charts

Usage:
    pip install scikit-learn numpy matplotlib
    python iris_svm.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

CLASSES   = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
COLORS    = ['#2a78d6', '#1baf7a', '#eda100']
ICONS     = ['setosa', 'versicolor', 'virginica']
FEAT_COLS = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
RANGES    = [(4.0, 8.0), (2.0, 4.5), (1.0, 7.0), (0.1, 2.5)]

IRIS_DATA = [
    [5.1,3.5,1.4,0.2,'Iris-setosa'],[4.9,3.0,1.4,0.2,'Iris-setosa'],[4.7,3.2,1.3,0.2,'Iris-setosa'],
    [4.6,3.1,1.5,0.2,'Iris-setosa'],[5.0,3.6,1.4,0.2,'Iris-setosa'],[5.4,3.9,1.7,0.4,'Iris-setosa'],
    [4.6,3.4,1.4,0.3,'Iris-setosa'],[5.0,3.4,1.5,0.2,'Iris-setosa'],[4.4,2.9,1.4,0.2,'Iris-setosa'],
    [4.9,3.1,1.5,0.1,'Iris-setosa'],[5.4,3.7,1.5,0.2,'Iris-setosa'],[4.8,3.4,1.6,0.2,'Iris-setosa'],
    [4.8,3.0,1.4,0.1,'Iris-setosa'],[4.3,3.0,1.1,0.1,'Iris-setosa'],[5.8,4.0,1.2,0.2,'Iris-setosa'],
    [5.7,4.4,1.5,0.4,'Iris-setosa'],[5.4,3.9,1.3,0.4,'Iris-setosa'],[5.1,3.5,1.4,0.3,'Iris-setosa'],
    [5.7,3.8,1.7,0.3,'Iris-setosa'],[5.1,3.8,1.5,0.3,'Iris-setosa'],[5.4,3.4,1.7,0.2,'Iris-setosa'],
    [5.1,3.7,1.5,0.4,'Iris-setosa'],[4.6,3.6,1.0,0.2,'Iris-setosa'],[5.1,3.3,1.7,0.5,'Iris-setosa'],
    [4.8,3.4,1.9,0.2,'Iris-setosa'],[5.0,3.0,1.6,0.2,'Iris-setosa'],[5.0,3.4,1.6,0.4,'Iris-setosa'],
    [5.2,3.5,1.5,0.2,'Iris-setosa'],[5.2,3.4,1.4,0.2,'Iris-setosa'],[4.7,3.2,1.6,0.2,'Iris-setosa'],
    [4.8,3.1,1.6,0.2,'Iris-setosa'],[5.4,3.4,1.5,0.4,'Iris-setosa'],[5.2,4.1,1.5,0.1,'Iris-setosa'],
    [5.5,4.2,1.4,0.2,'Iris-setosa'],[4.9,3.1,1.5,0.2,'Iris-setosa'],[5.0,3.2,1.2,0.2,'Iris-setosa'],
    [5.5,3.5,1.3,0.2,'Iris-setosa'],[4.9,3.6,1.4,0.1,'Iris-setosa'],[4.4,3.0,1.3,0.2,'Iris-setosa'],
    [5.1,3.4,1.5,0.2,'Iris-setosa'],[5.0,3.5,1.3,0.3,'Iris-setosa'],[4.5,2.3,1.3,0.3,'Iris-setosa'],
    [4.4,3.2,1.3,0.2,'Iris-setosa'],[5.0,3.5,1.6,0.6,'Iris-setosa'],[5.1,3.8,1.9,0.4,'Iris-setosa'],
    [4.8,3.0,1.4,0.3,'Iris-setosa'],[5.1,3.8,1.6,0.2,'Iris-setosa'],[4.6,3.2,1.4,0.2,'Iris-setosa'],
    [5.3,3.7,1.5,0.2,'Iris-setosa'],[5.0,3.3,1.4,0.2,'Iris-setosa'],
    [7.0,3.2,4.7,1.4,'Iris-versicolor'],[6.4,3.2,4.5,1.5,'Iris-versicolor'],[6.9,3.1,4.9,1.5,'Iris-versicolor'],
    [5.5,2.3,4.0,1.3,'Iris-versicolor'],[6.5,2.8,4.6,1.5,'Iris-versicolor'],[5.7,2.8,4.5,1.3,'Iris-versicolor'],
    [6.3,3.3,4.7,1.6,'Iris-versicolor'],[4.9,2.4,3.3,1.0,'Iris-versicolor'],[6.6,2.9,4.6,1.3,'Iris-versicolor'],
    [5.2,2.7,3.9,1.4,'Iris-versicolor'],[5.0,2.0,3.5,1.0,'Iris-versicolor'],[5.9,3.0,4.2,1.5,'Iris-versicolor'],
    [6.0,2.2,4.0,1.0,'Iris-versicolor'],[6.1,2.9,4.7,1.4,'Iris-versicolor'],[5.6,2.9,3.6,1.3,'Iris-versicolor'],
    [6.7,3.1,4.4,1.4,'Iris-versicolor'],[5.6,3.0,4.5,1.5,'Iris-versicolor'],[5.8,2.7,4.1,1.0,'Iris-versicolor'],
    [6.2,2.2,4.5,1.5,'Iris-versicolor'],[5.6,2.5,3.9,1.1,'Iris-versicolor'],[5.9,3.2,4.8,1.8,'Iris-versicolor'],
    [6.1,2.8,4.0,1.3,'Iris-versicolor'],[6.3,2.5,4.9,1.5,'Iris-versicolor'],[6.1,2.8,4.7,1.2,'Iris-versicolor'],
    [6.4,2.9,4.3,1.3,'Iris-versicolor'],[6.6,3.0,4.4,1.4,'Iris-versicolor'],[6.8,2.8,4.8,1.4,'Iris-versicolor'],
    [6.7,3.0,5.0,1.7,'Iris-versicolor'],[6.0,2.9,4.5,1.5,'Iris-versicolor'],[5.7,2.6,3.5,1.0,'Iris-versicolor'],
    [5.5,2.4,3.8,1.1,'Iris-versicolor'],[5.5,2.4,3.7,1.0,'Iris-versicolor'],[5.8,2.7,3.9,1.2,'Iris-versicolor'],
    [6.0,2.7,5.1,1.6,'Iris-versicolor'],[5.4,3.0,4.5,1.5,'Iris-versicolor'],[6.0,3.4,4.5,1.6,'Iris-versicolor'],
    [6.7,3.1,4.7,1.5,'Iris-versicolor'],[6.3,2.3,4.4,1.3,'Iris-versicolor'],[5.6,3.0,4.1,1.3,'Iris-versicolor'],
    [5.5,2.5,4.0,1.3,'Iris-versicolor'],[5.5,2.6,4.4,1.2,'Iris-versicolor'],[6.1,3.0,4.6,1.4,'Iris-versicolor'],
    [5.8,2.6,4.0,1.2,'Iris-versicolor'],[5.0,2.3,3.3,1.0,'Iris-versicolor'],[5.6,2.7,4.2,1.3,'Iris-versicolor'],
    [5.7,3.0,4.2,1.2,'Iris-versicolor'],[5.7,2.9,4.2,1.3,'Iris-versicolor'],[6.2,2.9,4.3,1.3,'Iris-versicolor'],
    [5.1,2.5,3.0,1.1,'Iris-versicolor'],[5.7,2.8,4.1,1.3,'Iris-versicolor'],
    [6.3,3.3,6.0,2.5,'Iris-virginica'],[5.8,2.7,5.1,1.9,'Iris-virginica'],[7.1,3.0,5.9,2.1,'Iris-virginica'],
    [6.3,2.9,5.6,1.8,'Iris-virginica'],[6.5,3.0,5.8,2.2,'Iris-virginica'],[7.6,3.0,6.6,2.1,'Iris-virginica'],
    [4.9,2.5,4.5,1.7,'Iris-virginica'],[7.3,2.9,6.3,1.8,'Iris-virginica'],[6.7,2.5,5.8,1.8,'Iris-virginica'],
    [7.2,3.6,6.1,2.5,'Iris-virginica'],[6.5,3.2,5.1,2.0,'Iris-virginica'],[6.4,2.7,5.3,1.9,'Iris-virginica'],
    [6.8,3.0,5.5,2.1,'Iris-virginica'],[5.7,2.5,5.0,2.0,'Iris-virginica'],[5.8,2.8,5.1,2.4,'Iris-virginica'],
    [6.4,3.2,5.3,2.3,'Iris-virginica'],[6.5,3.0,5.5,1.8,'Iris-virginica'],[7.7,3.8,6.7,2.2,'Iris-virginica'],
    [7.7,2.6,6.9,2.3,'Iris-virginica'],[6.0,2.2,5.0,1.5,'Iris-virginica'],[6.9,3.2,5.7,2.3,'Iris-virginica'],
    [5.6,2.8,4.9,2.0,'Iris-virginica'],[7.7,2.8,6.7,2.0,'Iris-virginica'],[6.3,2.7,4.9,1.8,'Iris-virginica'],
    [6.7,3.3,5.7,2.1,'Iris-virginica'],[7.2,3.2,6.0,1.8,'Iris-virginica'],[6.2,2.8,4.8,1.8,'Iris-virginica'],
    [6.1,3.0,4.9,1.8,'Iris-virginica'],[6.4,2.8,5.6,2.1,'Iris-virginica'],[7.2,3.0,5.8,1.6,'Iris-virginica'],
    [7.4,2.8,6.1,1.9,'Iris-virginica'],[7.9,3.8,6.4,2.0,'Iris-virginica'],[6.4,2.8,5.6,2.2,'Iris-virginica'],
    [6.3,2.8,5.1,1.5,'Iris-virginica'],[6.1,2.6,5.6,1.4,'Iris-virginica'],[7.7,3.0,6.1,2.3,'Iris-virginica'],
    [6.3,3.4,5.6,2.4,'Iris-virginica'],[6.4,3.1,5.5,1.8,'Iris-virginica'],[6.0,3.0,4.8,1.8,'Iris-virginica'],
    [6.9,3.1,5.4,2.1,'Iris-virginica'],[6.7,3.1,5.6,2.4,'Iris-virginica'],[6.9,3.1,5.1,2.3,'Iris-virginica'],
    [5.8,2.7,5.1,1.9,'Iris-virginica'],[6.8,3.2,5.9,2.3,'Iris-virginica'],[6.7,3.3,5.7,2.5,'Iris-virginica'],
    [6.7,3.0,5.2,2.3,'Iris-virginica'],[6.3,2.5,5.0,1.9,'Iris-virginica'],[6.5,3.0,5.2,2.0,'Iris-virginica'],
    [6.2,3.4,5.4,2.3,'Iris-virginica'],[5.9,3.0,5.1,1.8,'Iris-virginica'],
]

# ── Train SVM model ──────────────────────────────────────────────────────────
X = np.array([row[:4] for row in IRIS_DATA])
y_raw = np.array([row[4] for row in IRIS_DATA])

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)
y = label_encoder.transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_scaled, y)


def predict(features):
    """SVM prediction with class probabilities."""
    feats_scaled = scaler.transform([features])
    pred_encoded = model.predict(feats_scaled)[0]
    proba = model.predict_proba(feats_scaled)[0]
    predicted = label_encoder.inverse_transform([pred_encoded])[0]
    probs = {cls: p for cls, p in zip(label_encoder.classes_, proba)}
    return predicted, probs


def get_input():
    """Prompt user to enter four measurements with validation."""
    prompts = [
        ("Sepal length", 4.0, 8.0),
        ("Sepal width",  2.0, 4.5),
        ("Petal length", 1.0, 7.0),
        ("Petal width",  0.1, 2.5),
    ]
    values = []
    print("\n" + "=" * 50)
    print("  IRIS SPECIES PREDICTOR — SVM")
    print("=" * 50)
    print("  Enter measurements (in cm) to classify.\n")

    for name, lo, hi in prompts:
        while True:
            try:
                val = float(input(f"  {name} ({lo}-{hi} cm): "))
                if lo <= val <= hi:
                    values.append(val)
                    break
                else:
                    print(f"  x  Value out of range. Must be between {lo} and {hi}.")
            except ValueError:
                print("  x  Please enter a number.")

    return values


def visualize(features, predicted, probs):
    """Create a 4-panel visualization including the SVM decision boundary."""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#fafaf9')
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    pred_idx = CLASSES.index(predicted)
    pred_col = COLORS[pred_idx]

    # ── Panel 1: Probability bar chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#fafaf9')
    class_names = [c.replace('Iris-', '') for c in CLASSES]
    pct_vals = [probs[c] * 100 for c in CLASSES]
    bars = ax1.barh(class_names, pct_vals, color=COLORS, height=0.5, edgecolor='none')
    for bar, pct in zip(bars, pct_vals):
        ax1.text(min(pct + 1.5, 97), bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va='center', ha='left', fontsize=11, fontweight='500',
                 color='#2c2c2a')
    ax1.set_xlim(0, 108)
    ax1.set_xlabel('Probability (%)', fontsize=10, color='#898781')
    ax1.set_title('SVM prediction probabilities', fontsize=12, fontweight='500',
                  color='#0b0b0b', pad=10)
    ax1.tick_params(colors='#898781', labelsize=10)
    ax1.spines[['top', 'right', 'bottom']].set_visible(False)
    ax1.spines['left'].set_color('#e1e0d9')
    ax1.xaxis.grid(True, color='#e1e0d9', linewidth=0.7)
    ax1.set_axisbelow(True)

    # ── Panel 2: SVM decision boundary (petal length vs petal width) ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#fafaf9')

    X_2d = X[:, [2, 3]]
    scaler_2d = StandardScaler().fit(X_2d)
    X_2d_scaled = scaler_2d.transform(X_2d)
    model_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model_2d.fit(X_2d_scaled, y)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.3, X_2d[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_scaled = scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model_2d.predict(grid_scaled).reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#E6F1FB', '#E1F5EE', '#FAEEDA'])
    ax2.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_light)

    for i, cls in enumerate(CLASSES):
        mask = y_raw == cls
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=COLORS[i], s=25,
                   edgecolor='white', linewidth=0.5, label=cls.replace('Iris-', ''))

    ax2.scatter(features[2], features[3], c='#e34948', s=140, marker='*',
               edgecolor='white', linewidth=1, zorder=5, label='Your input')
    ax2.set_xlabel('Petal length (cm)', fontsize=10, color='#898781')
    ax2.set_ylabel('Petal width (cm)', fontsize=10, color='#898781')
    ax2.set_title('SVM decision boundary\n(petal length vs width)', fontsize=12,
                  fontweight='500', color='#0b0b0b', pad=10)
    ax2.tick_params(colors='#898781', labelsize=9)
    ax2.legend(fontsize=8, frameon=False, loc='upper left')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color('#e1e0d9')

    # ── Panel 3: Feature value bars vs class means ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#fafaf9')
    n_feat = len(FEAT_COLS)
    x_pos = np.arange(n_feat)
    width = 0.2

    class_means = {cls: X[y_raw == cls].mean(axis=0) for cls in CLASSES}
    for i, (cls, col) in enumerate(zip(CLASSES, COLORS)):
        ax3.bar(x_pos + i * width, class_means[cls], width, color=col, alpha=0.75,
               label=cls.replace('Iris-', ''), edgecolor='none')

    ax3.bar(x_pos + 3 * width, features, width, color='#e34948', alpha=0.9,
           label='Your input', edgecolor='none')
    ax3.set_xticks(x_pos + 1.5 * width)
    ax3.set_xticklabels(FEAT_COLS, fontsize=9, color='#5f5e5a')
    ax3.set_ylabel('Value (cm)', fontsize=10, color='#898781')
    ax3.set_title('Your values vs class means', fontsize=12, fontweight='500',
                  color='#0b0b0b', pad=10)
    ax3.tick_params(colors='#898781', labelsize=9)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.spines[['left', 'bottom']].set_color('#e1e0d9')
    ax3.yaxis.grid(True, color='#e1e0d9', linewidth=0.7)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, frameon=False, ncol=2, loc='upper left')

    # ── Panel 4: Result summary ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(pred_col + '18')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    ax4.text(0.5, 0.88, 'Predicted species (SVM)', ha='center', va='center',
             fontsize=11, color='#5f5e5a')
    ax4.text(0.5, 0.66, predicted, ha='center', va='center',
             fontsize=18, fontweight='500', color=pred_col)
    ax4.text(0.5, 0.50, f"Confidence: {probs[predicted]*100:.1f}%",
             ha='center', va='center', fontsize=12, color='#5f5e5a')

    ax4.add_patch(mpatches.FancyBboxPatch((0.05, 0.08), 0.9, 0.30,
        boxstyle='round,pad=0.02', linewidth=0, facecolor='white', alpha=0.6))

    feat_summary = (
        f"Sepal: {features[0]:.1f} x {features[1]:.1f} cm\n"
        f"Petal: {features[2]:.1f} x {features[3]:.1f} cm\n"
        f"Kernel: RBF   C: 1.0"
    )
    ax4.text(0.5, 0.23, feat_summary, ha='center', va='center',
             fontsize=11, color='#2c2c2a', linespacing=1.8)

    fig.suptitle('Iris Species Classifier — Support Vector Machine (SVM)',
                 fontsize=14, fontweight='500', color='#0b0b0b', y=0.98)

    plt.savefig('iris_svm_prediction.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\n  Chart saved -> iris_svm_prediction.png")
    plt.show()


def main():
    features = get_input()
    predicted, probs = predict(features)

    print("\n" + "-" * 50)
    print(f"  Predicted species : {predicted}")
    print(f"  Confidence        : {probs[predicted]*100:.1f}%")
    print("-" * 50)
    print("  Class probabilities:")
    for cls in CLASSES:
        bar = "#" * int(probs[cls] * 30)
        print(f"  {cls:<20} {bar:<30} {probs[cls]*100:5.1f}%")
    print("-" * 50)
    print("\n  Generating visualisation...")

    visualize(features, predicted, probs)

    again = input("\n  Predict another flower? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("\n  Done. Goodbye!\n")


if __name__ == "__main__":
    main()