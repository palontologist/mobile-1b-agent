package com.ai.edge.agent;

import com.google.ai.edge.litertlm.InternalJsonTool;
import com.google.ai.edge.litertlm.ToolProvider;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Registers the assistant's offline tools with LiteRT-LM, for real.
 *
 * <p>This is Java on purpose. {@code ToolProvider}'s only abstract method is
 * name-mangled by the Kotlin compiler ({@code provideTools$third_party_...}), and
 * Kotlin refuses to override it: the declaration is flagged "invisible", which is
 * how the compiler marks a member it will not let you implement. At the JVM level
 * that is just a method with an unusual name, and Java can override it without
 * complaint.
 *
 * <p>The cost of this file is that it depends on an internal API with no
 * compatibility guarantee. If upstream renames the method this stops compiling,
 * which is the failure mode worth having: loud and immediate, rather than a
 * silently empty tool list that reads as "the model is bad at tools".
 */
public final class BenchTools {

  private BenchTools() {}

  /**
   * Tool name to [description, paramName:paramDescription...].
   *
   * <p>Weather and web search are absent on purpose. They cannot be served offline,
   * so including them would measure whether the model reaches for tools the
   * assistant would have to refuse anyway.
   */
  private static final String[][] SPECS = {
    {"get_current_time", "Get the current date and time"},
    {"get_location", "Get the device's GPS coordinates"},
    {
      "search_conversations",
      "Search recorded conversations for what someone said",
      "query:what to search for",
      "speaker:who said it"
    },
    {"list_tasks", "List tasks and reminders stored on the device"},
    {"add_task", "Create a task", "title:what the task is"},
    {
      "add_reminder",
      "Create a reminder",
      "title:what to be reminded of",
      "when:when it is due"
    },
    {"set_timer", "Start a countdown timer", "minutes:how many minutes"},
  };

  /** One tool, shaped like ReflectionTool shapes its own: OpenAI-style name/description/parameters. */
  static final class Tool implements InternalJsonTool {
    private final String name;
    private final String description;
    /** Each entry is {@code name:description}. */
    private final String[] params;

    Tool(String name, String description, String[] params) {
      this.name = name;
      this.description = description;
      this.params = params;
    }

    String name() {
      return name;
    }

    @Override
    public JsonObject getToolDescription() {
      JsonObject properties = new JsonObject();
      for (String param : params) {
        int colon = param.indexOf(':');
        JsonObject prop = new JsonObject();
        prop.addProperty("type", "string");
        prop.addProperty("description", param.substring(colon + 1));
        properties.add(param.substring(0, colon), prop);
      }

      JsonObject schema = new JsonObject();
      schema.addProperty("type", "object");
      schema.add("properties", properties);
      schema.add("required", new JsonArray());

      JsonObject tool = new JsonObject();
      tool.addProperty("name", name);
      tool.addProperty("description", description);
      tool.add("parameters", schema);
      return tool;
    }

    /**
     * Unreachable in the benchmark: automaticToolCalling is off, so the model only
     * has to name a tool. Returns an empty string rather than throwing so that
     * flipping automatic tool calling on does not immediately crash.
     */
    @Override
    public Object execute(JsonObject args) {
      return "";
    }
  }

  /**
   * The tool contract FunctionGemma 270M was actually fine-tuned on.
   *
   * <p>Added after measuring 0/15 against a home-grown action set. The hypothesis
   * worth testing is that the model learned a fixed prompt contract rather than
   * general tool selection: mobile-actions is a fine-tune of functiongemma-270m-it
   * over one specific action set, so offering a different set of names and
   * descriptions may be asking it to generalize where it was never trained to.
   *
   * <p>Names and count follow the set described in litert-samples#349 (flashlight,
   * calendar, photo, alarm, message, note, no-op), which is where the 28% figure came
   * from. If this scores well where the other set scored zero, the earlier result was
   * measuring contract mismatch and not model capability.
   */
  private static final String[][] MOBILE_SPECS = {
    {"turn_on_flashlight", "Turns on the phone flashlight"},
    {"turn_off_flashlight", "Turns off the phone flashlight"},
    {"get_calendar_events", "Lists the events on the user's calendar"},
    {"take_photo", "Takes a photo with the camera"},
    {"set_alarm", "Sets an alarm"},
    {"send_message", "Sends a text message"},
    {"create_note", "Creates a note"},
    {"no_op", "Does nothing"},
  };

  public static final class MobileProvider extends ToolProvider {
    @Override
    public Map<String, InternalJsonTool>
        provideTools$third_party_odml_litert_lm_kotlin_java_com_google_ai_edge_litertlm_litertlm_android() {
      Map<String, InternalJsonTool> out = new LinkedHashMap<>();
      for (String[] spec : MOBILE_SPECS) {
        out.put(spec[0], new Tool(spec[0], spec[1], new String[0]));
      }
      return out;
    }
  }

  public static final class Provider extends ToolProvider {
    @Override
    public Map<String, InternalJsonTool>
        provideTools$third_party_odml_litert_lm_kotlin_java_com_google_ai_edge_litertlm_litertlm_android() {
      Map<String, InternalJsonTool> out = new LinkedHashMap<>();
      for (String[] spec : SPECS) {
        out.put(spec[0], new Tool(spec[0], spec[1], java.util.Arrays.copyOfRange(spec, 2, spec.length)));
      }
      return out;
    }
  }
}