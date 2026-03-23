import { FormEvent, useEffect, useState } from "react";

type Product = {
  id: string;
  name: string;
  description: string;
  category: string;
  price: number;
  stock: number;
};

type Message = {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  products?: Product[];
  visibleCount?: number;
};

type CartItem = {
  id: string;
  quantity: number;
  product: Product;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function App() {
  const [userId] = useState("demo-user");
  const [view, setView] = useState<"chat" | "cart" | "wishlist">("chat");

  const [orderPlaced, setOrderPlaced] = useState(false);
  const [finalAmount, setFinalAmount] = useState(0);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "Ask for products in natural language. Example: cheap running shoes under 1500.",
    },
  ]);

  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [wishlist, setWishlist] = useState<CartItem[]>([]);
  const [cartCount, setCartCount] = useState(0);

  const totalAmount = cartItems.reduce(
    (sum, item) => sum + item.product.price * item.quantity,
    0
  );

  useEffect(() => {
    refreshCart();
    refreshWishlist();
  }, []);
  async function refreshSearchResults() {
  setMessages((msgs) =>
    msgs.map((m) => {
      if (!m.products) return m;

      return {
        ...m,
        products: m.products.map((p) => ({
          ...p,
          stock: 0, // temporary until next search
        })),
      };
    })
  );
}
  async function refreshCart() {
    const res = await fetch(`${API_URL}/cart?user_id=${userId}`);
    if (!res.ok) return;

    const data = await res.json();
    setCartItems(data.items);

    const total = data.items.reduce(
      (s: number, i: CartItem) => s + i.quantity,
      0
    );
    setCartCount(total);
  }

  async function refreshWishlist() {
    const res = await fetch(`${API_URL}/wishlist?user_id=${userId}`);
    if (!res.ok) return;

    const data = await res.json();
    setWishlist(data.items);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: input,
    };

    setMessages((c) => [...c, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/search?q=${input}`);
      const data = await res.json();

      setMessages((c) => [
        ...c,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `Found ${data.count} results`,
          products: data.results,
          visibleCount: 6,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function loadMore(messageId: string) {
    setMessages((msgs) =>
      msgs.map((m) =>
        m.id === messageId
          ? { ...m, visibleCount: (m.visibleCount ?? 6) + 6 }
          : m
      )
    );
  }

  async function handleAction(productId: string, mode: "cart" | "wishlist") {
    const endpoint = mode === "cart" ? "/cart/add" : "/wishlist/add";

    await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        product_id: productId,
        quantity: 1,
      }),
    });

    if (mode === "cart") refreshCart();
    if (mode === "wishlist") refreshWishlist();
  }

  async function removeFromCart(productId: string) {
    await fetch(`${API_URL}/cart/remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, product_id: productId }),
    });

    refreshCart();
  }

  async function updateQty(productId: string, change: number) {
    await fetch(`${API_URL}/cart/update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        product_id: productId,
        quantity: change,
      }),
    });

    refreshCart();
  }

  async function removeFromWishlist(productId: string) {
    await fetch(`${API_URL}/wishlist/remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, product_id: productId }),
    });

    refreshWishlist();
  }

  // ✅ FIXED: now calls backend checkout
  async function handleCheckout() {
    if (cartItems.length === 0) return;

    const res = await fetch(
      `${API_URL}/cart/checkout?user_id=${userId}`,
      { method: "POST" }
    );

    if (!res.ok) return;

    setFinalAmount(totalAmount);
    setOrderPlaced(true);
    setCartItems([]);
    setCartCount(0);
    await refreshSearchResults();
  }

  return (
    <div className="min-h-screen px-4 py-8 text-ink">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 lg:flex-row">
        <section className="flex-1 overflow-hidden rounded-[28px] border border-white/60 bg-white/80 shadow-soft backdrop-blur">

          {/* HEADER */}
          <div className="border-b border-ink/10 bg-ink px-6 py-5 text-white">
            <div className="flex items-center justify-between">
              <div className="flex gap-3 items-center">
                {view !== "chat" && (
                  <button onClick={() => setView("chat")} className="text-xs underline">
                    Back
                  </button>
                )}
                <h1 className="text-2xl font-semibold">AI Shopping Assistant</h1>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setView("cart");
                    setOrderPlaced(false);
                  }}
                  className="rounded-full bg-white/10 px-3 py-1 text-sm"
                >
                  🛒 {cartCount}
                </button>

                <button
                  onClick={() => setView("wishlist")}
                  className="rounded-full bg-white/10 px-3 py-1 text-sm"
                >
                  ❤️ {wishlist.length}
                </button>
              </div>
            </div>
          </div>

          <div className="flex h-[70vh] flex-col">
            <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6">

              {/* CHAT */}
              {view === "chat" && (
                <div className="space-y-4">
                  {messages.map((m) => (
                    <div
                      key={m.id}
                      className={`rounded-3xl px-4 py-3 ${
                        m.role === "user" ? "bg-blue-100" : "bg-sand/60"
                      }`}
                    >
                      <p>{m.text}</p>

                      {m.products && (
                        <>
                          <div className="mt-4 grid gap-3 md:grid-cols-2">
                            {m.products
                              .slice(0, m.visibleCount ?? 6)
                              .map((p) => {
                                const cartItem = cartItems.find(
                                  (c) => c.product.id === p.id
                                );
                                const alreadyInCart = cartItem
                                  ? cartItem.quantity
                                  : 0;

                                const isOutOfStock =
                                  p.stock === 0 ||
                                  alreadyInCart >= p.stock;

                                return (
                                  <article key={p.id} className="rounded-2xl border p-4 bg-white">
                                    <h2 className="font-semibold">{p.name}</h2>
                                    <p className="text-sm">{p.description}</p>
                                    <p className="font-medium">₹ {p.price}</p>

                                    <div className="mt-3 flex gap-2">
                                      <button
                                        onClick={() => handleAction(p.id, "cart")}
                                        disabled={isOutOfStock}
                                        className="bg-ink text-white px-3 py-1 rounded disabled:opacity-50"
                                      >
                                        {isOutOfStock ? "Out of Stock" : "Add to Cart"}
                                      </button>

                                      <button
                                        onClick={() => handleAction(p.id, "wishlist")}
                                        className="border px-3 py-1 rounded"
                                      >
                                        Wishlist
                                      </button>
                                    </div>
                                  </article>
                                );
                              })}
                          </div>

                          {m.visibleCount! < m.products.length && (
                            <button
                              onClick={() => loadMore(m.id)}
                              className="mt-3 text-sm underline"
                            >
                              Load more
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* CART */}
              {view === "cart" && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Cart</h2>

                  {orderPlaced && (
                    <div className="bg-green-100 border border-green-300 p-6 rounded text-center mb-4">
                      <div className="text-4xl mb-2">✅</div>
                      <h3 className="text-lg font-semibold">Order Placed!</h3>
                      <p>Total Paid: ₹ {finalAmount.toFixed(2)}</p>
                    </div>
                  )}

                  {!orderPlaced && (
                    <>
                      {cartItems.map((item) => {
                        const itemTotal =
                          item.product.price * item.quantity;

                        return (
                          <div key={item.product.id} className="border p-3 mb-2">
                            <h3>{item.product.name}</h3>

                            <p className="text-sm text-gray-600">
                              Price: ₹ {item.product.price}
                            </p>

                            <p className="text-sm font-medium">
                              Subtotal: ₹ {itemTotal.toFixed(2)}
                            </p>

                            <div className="flex items-center gap-2 mt-2">
                              <button
                                onClick={() => updateQty(item.product.id, -1)}
                                className="px-2 bg-gray-200 rounded"
                              >
                                -
                              </button>

                              <span>{item.quantity}</span>

                              <button
                                onClick={() => updateQty(item.product.id, 1)}
                                className="px-2 bg-gray-200 rounded"
                              >
                                +
                              </button>
                            </div>

                            <button
                              onClick={() => removeFromCart(item.product.id)}
                              className="text-red-500 mt-2"
                            >
                              Remove
                            </button>
                          </div>
                        );
                      })}

                      <div className="mt-4 border-t pt-4 flex justify-between items-center">
                        <h3 className="text-lg font-semibold">
                          Total: ₹ {totalAmount.toFixed(2)}
                        </h3>

                        <button
                          onClick={handleCheckout}
                          className="bg-green-600 text-white px-4 py-2 rounded"
                        >
                          Checkout
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* WISHLIST */}
              {view === "wishlist" && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Wishlist</h2>

                  {wishlist.map((item) => (
                    <div key={item.product.id} className="border p-3 mb-2">
                      <h3>{item.product.name}</h3>

                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => handleAction(item.product.id, "cart")}
                          className="bg-ink text-white px-2 py-1 rounded"
                        >
                          Move to Cart
                        </button>

                        <button
                          onClick={() => removeFromWishlist(item.product.id)}
                          className="text-red-500"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* INPUT */}
            {view === "chat" && (
              <form onSubmit={handleSubmit} className="border-t p-4">
                <div className="flex gap-2">
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className="flex-1 border px-3 py-2 rounded"
                  />
                  <button className="bg-ember text-white px-4 rounded">
                    {loading ? "..." : "Send"}
                  </button>
                </div>
              </form>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;